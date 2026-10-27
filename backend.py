"""LungLens chest X-ray API. Run with python -m uvicorn backend:app."""
import logging
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from lunglens.config import Settings
from lunglens.classifier import Classifier, ModelUnavailable
from lunglens.explainer import Explainer, ExplanationUnavailable
from lunglens.imaging import decode_input, MAX_BYTES
from lunglens.quality import inspect_quality
from lunglens.anatomy import Anatomy
from lunglens.visuals import png_base64, preview_image
from lunglens.evaluation import evaluate_observations
from pydantic import ValidationError
from lunglens.schemas import AnalysisResponse, HeatmapResponse, AnatomyResponse, EvaluationRequest

settings = Settings.from_env()
classifier = Classifier(settings)
explainer = Explainer(settings)
anatomy = Anatomy(settings)
app = FastAPI(title='LungLens Chest X-ray Backend',version='2.1.0',description='Research chest X-ray finding scores with optional MedGemma explanations. Upload frontal AP/PA images; no annotations needed.')
if settings.allowed_origins:
    app.add_middleware(CORSMiddleware,allow_origins=list(settings.allowed_origins),allow_methods=['GET','POST'],allow_headers=['Content-Type'])

@app.get('/health')
def health():
    return {'project':'LungLens','status':'ok','classifier':classifier.info(),'medgemma':explainer.info(),'anatomy':anatomy.info()}

@app.get('/models')
def models():
    return {'classifier':classifier.info(),'medgemma':explainer.info(),'anatomy':anatomy.info()}

@app.get('/ready')
def ready():
    try:
        return {'ready':True,'classifier':classifier.ready(),'medgemma_required':False}
    except ModelUnavailable as exc:
        raise HTTPException(503,str(exc)) from None

@app.get('/medgemma/ready')
def medgemma_ready():
    try:
        return explainer.ready()
    except ExplanationUnavailable as exc:
        raise HTTPException(503,str(exc)) from None

@app.post('/predict',response_model=AnalysisResponse,response_model_exclude_none=True,openapi_extra={'requestBody':{'required':True,'content':{
    'image/png':{'schema':{'type':'string','format':'binary'}},
    'image/jpeg':{'schema':{'type':'string','format':'binary'}},
    'application/dicom':{'schema':{'type':'string','format':'binary'}},
}}})
async def predict(request: Request, view: Literal['AP','PA'] | None = None, explain: bool = False):
    image, metadata = await uploaded_image(request, view)
    try:
        result = await run_in_threadpool(classifier.predict,image,view or metadata.get('recorded_view'))
    except ModelUnavailable as exc:
        raise HTTPException(503,str(exc)) from None
    except Exception:
        logging.exception('Classifier inference failed')
        raise HTTPException(503,'Classifier could not complete inference') from None
    result['input_metadata'] = metadata
    result['quality'] = await run_in_threadpool(inspect_quality,image)
    result['explanation'] = await run_in_threadpool(explainer.explain,result) if explain else {'status':'not_requested'}
    return result


async def read_body(request):
    body = bytearray()
    async for chunk in request.stream():
        if len(body)+len(chunk) > MAX_BYTES:
            raise HTTPException(413,'Request exceeds the 20 MB limit')
        body.extend(chunk)
    return bytes(body)


async def uploaded_image(request, view=None):
    body = await read_body(request)
    try:
        image, metadata = await run_in_threadpool(decode_input,body,request.headers.get('content-type',''))
        recorded = metadata.get('recorded_view')
        if view and recorded and view != recorded:
            raise ValueError('Declared view conflicts with DICOM ViewPosition; use the recorded AP/PA view')
        return image, metadata
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from None


IMAGE_BODY = {'requestBody':{'required':True,'content':{
    mime:{'schema':{'type':'string','format':'binary'}}
    for mime in ('image/png','image/jpeg','application/dicom')
}}}


@app.get('/research/capabilities')
def research_capabilities():
    import torchxrayvision as xrv
    metadata = xrv.models.model_urls[settings.weights]
    return {'version':'1.0','findings':metadata['labels'],
            'classifier':classifier.info(),'anatomy':anatomy.info(),
            'features':{'gradcam':True,'anatomical_masks':True,'quality_flags':True,
                        'dicom_cr_dx':True,'aggregate_evaluation':True,'medsiglip':False},
            'visualization_max_side':1024,'max_upload_bytes':MAX_BYTES,
            'reference_thresholds':dict(zip(metadata['labels'],metadata['op_threshs'])),
            'notice':'Research capabilities; availability is not clinical validation. Segmentation requires its separately installed checkpoint.'}


@app.get('/anatomy/ready')
def anatomy_ready():
    try:
        return {'ready':True,'anatomy':anatomy.ready()}
    except ModelUnavailable as exc:
        raise HTTPException(503,str(exc)) from None


@app.post('/research/inspect',openapi_extra=IMAGE_BODY)
async def inspect_input(request: Request):
    image, metadata = await uploaded_image(request)
    quality = await run_in_threadpool(inspect_quality,image)
    preview = await run_in_threadpool(preview_image,image)
    encoded = await run_in_threadpool(png_base64,preview)
    return {'width':image.width,'height':image.height,'input_metadata':metadata,
            'preview_width':preview.width,'preview_height':preview.height,
            'preview_png_base64':encoded,'quality':quality,
            'notice':'Input inspection does not verify chest anatomy or diagnostic quality. Preview pixels may contain burned-in identifiers.'}


@app.post('/research/heatmap',response_model=HeatmapResponse,openapi_extra=IMAGE_BODY)
async def research_heatmap(request: Request, view: Literal['AP','PA'] | None = None, finding: str = Query(min_length=1,max_length=80)):
    image, metadata = await uploaded_image(request,view)
    try:
        return await run_in_threadpool(classifier.heatmap,image,view or metadata.get('recorded_view'),finding)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from None
    except ModelUnavailable as exc:
        raise HTTPException(503,str(exc)) from None
    except Exception:
        logging.exception('Grad-CAM failed')
        raise HTTPException(503,'Heatmap could not be generated') from None


@app.post('/research/anatomy',response_model=AnatomyResponse,openapi_extra=IMAGE_BODY)
async def research_anatomy(request: Request, view: Literal['AP','PA'] | None = None, threshold: float = Query(default=.5,gt=0,lt=1,allow_inf_nan=False)):
    image, metadata = await uploaded_image(request,view)
    try:
        return await run_in_threadpool(anatomy.segment,image,threshold)
    except ModelUnavailable as exc:
        raise HTTPException(503,str(exc)) from None
    except Exception:
        logging.exception('Anatomy segmentation failed')
        raise HTTPException(503,'Anatomical masks could not be generated') from None


@app.post('/research/evaluate',openapi_extra={'requestBody':{'required':True,'content':{'application/json':{'schema':{
    'type':'object','required':['model_weights','checkpoint_sha256','observations'],
    'properties':{'model_weights':{'type':'string','enum':['densenet121-res224-all']},
                  'checkpoint_sha256':{'type':'string','pattern':'^[a-f0-9]{64}$'},
                  'observations':{'type':'array','minItems':1,'maxItems':50000,'items':{'type':'object',
                    'required':['case_id','label','truth','score'],'properties':{
                    'case_id':{'type':'string','description':'Pseudonymous case ID; never patient identifiers'},
                    'label':{'type':'string'},'truth':{'type':'integer','enum':[0,1]},
                    'score':{'type':'number','minimum':0,'maximum':1}}}}}
}}}}})
async def research_evaluate(request: Request):
    body = await read_body(request)
    try:
        payload = EvaluationRequest.model_validate_json(body)
    except ValidationError:
        raise HTTPException(422,'Supply a valid model ID, checkpoint SHA-256, and up to 50,000 observations with pseudonymous case_id, exact label, truth 0/1 and finite score 0..1') from None
    def calculate():
        import torchxrayvision as xrv
        metadata = xrv.models.model_urls[settings.weights]
        references = dict(zip(metadata['labels'],metadata['op_threshs']))
        return evaluate_observations(payload,references)
    try:
        return await run_in_threadpool(calculate)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from None
    except Exception:
        logging.exception('Aggregate evaluation failed')
        raise HTTPException(503,'Evaluation could not be completed; check evaluation dependencies') from None


# Keep API routes above this mount; hash-based frontend routes share the same origin.
frontend_dist = Path(__file__).resolve().parent / 'frontend' / 'dist'
if (frontend_dist / 'index.html').is_file():
    app.mount('/', StaticFiles(directory=str(frontend_dist), html=True), name='frontend')
