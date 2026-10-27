import hashlib
import json
import threading
import numpy as np
from PIL import Image
from .classifier import ModelUnavailable
from .imaging import preprocess
from .visuals import preview_image, project_map, png_base64, crop_box

WEIGHTS_URL = 'https://github.com/mlmed/torchxrayvision/releases/download/v1/pspnet_chestxray_best_model_4.pth'
STRUCTURES = ('Left Lung','Right Lung','Heart')
COLORS = {'Left Lung':(28,166,140), 'Right Lung':(47,130,202), 'Heart':(232,164,61)}


class Anatomy:
    def __init__(self, settings):
        self.settings = settings
        self.cache_dir = settings.cache_dir.parent / 'anatomy'
        self.model = None
        self.lock = threading.Lock()
        self.sha256 = None

    def info(self):
        return {'name':'ChestX-Det PSPNet', 'loaded':self.model is not None,
                'checkpoint_present':(self.cache_dir/WEIGHTS_URL.rsplit('/',1)[-1]).is_file(),
                'checkpoint_sha256':self.sha256, 'device':self.settings.device,
                'structures':list(STRUCTURES)}

    def _load(self):
        if self.model is not None:
            return
        path = self.cache_dir / WEIGHTS_URL.rsplit('/',1)[-1]
        manifest = self.cache_dir / 'checkpoint.json'
        if not path.is_file() or not manifest.is_file():
            raise ModelUnavailable('Run python scripts/setup_models.py --anatomy to install the anatomy checkpoint')
        try:
            import torch
            import torchxrayvision as xrv
            if self.settings.device == 'cuda' and not torch.cuda.is_available():
                raise ModelUnavailable('CUDA requested but unavailable')
            with path.open('rb') as checkpoint:
                digest = hashlib.file_digest(checkpoint,'sha256').hexdigest()
            record = json.loads(manifest.read_text())
            if record.get('sha256') != digest or record.get('source') != WEIGHTS_URL:
                raise ModelUnavailable('Anatomy checkpoint integrity check failed; rerun setup_models.py --anatomy')
            model = xrv.baseline_models.chestx_det.PSPNet(cache_dir=str(self.cache_dir)).eval().to(self.settings.device)
            if not set(STRUCTURES).issubset(model.targets):
                raise ModelUnavailable('Unexpected anatomical model targets')
            self.model, self.sha256 = model, digest
        except ModelUnavailable:
            raise
        except Exception as exc:
            raise ModelUnavailable('Anatomy model could not load; check its dependencies and cached checkpoint') from exc

    def ready(self):
        with self.lock:
            self._load()
        return self.info()

    def segment(self, image, threshold=.5):
        import torch
        import torchxrayvision as xrv
        with self.lock:
            self._load()
            tensor = torch.from_numpy(preprocess(image,xrv,size=512)[None]).to(self.settings.device)
            with torch.inference_mode():
                probabilities = torch.sigmoid(self.model(tensor))[0].cpu().numpy()
            if probabilities.shape != (14,512,512) or not np.isfinite(probabilities).all():
                raise ModelUnavailable('Anatomy model returned invalid output')
            preview = preview_image(image)
            combined = preview.convert('RGBA')
            structures = []
            for label in STRUCTURES:
                mask = (probabilities[self.model.targets.index(label)] >= threshold).astype(np.float32)
                gray, box = project_map(mask, image, preview, resample=Image.Resampling.NEAREST)
                binary = (np.asarray(gray) >= 128).astype('uint8')*255
                color = np.zeros((preview.height,preview.width,4),dtype='uint8')
                color[...,:3] = COLORS[label]
                color[...,3] = (binary > 0)*80
                combined = Image.alpha_composite(combined,Image.fromarray(color))
                structures.append({'label':label,'color_rgb':list(COLORS[label]),
                                   'mask_png_base64':png_base64(Image.fromarray(binary)),
                                   'crop_area_fraction':float(mask.mean())})
            return {'method':'ChestX-Det PSPNet anatomical segmentation', 'model':self.info(),
                    'threshold':threshold,'width':preview.width,'height':preview.height,
                    'crop_box':list(crop_box(image)),'preview_crop_box':list(box),
                    'structures':structures,'overlay_png_base64':png_base64(combined.convert('RGB')),
                    'notice':'Automatic anatomical research masks, not pathology boundaries. Masks use the center crop; outside-crop pixels are unassessed. Fractions refer to the model crop and are not diagnostic measurements.'}
