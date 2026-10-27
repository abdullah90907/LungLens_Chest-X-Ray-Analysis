"""Local synthetic calculations only; no HTTP requests or browser uploads."""
import base64
import io
import json
import unittest
import numpy as np
from PIL import Image
from lunglens.visuals import crop_box, preview_image, project_map
from lunglens.quality import inspect_quality
from lunglens.schemas import EvaluationRequest
from lunglens.evaluation import evaluate_observations
from lunglens.imaging import decode_input


def synthetic_dicom(frames=1):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, DigitalXRayImageStorageForPresentation, generate_uid
    from pydicom import dcmwrite
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = DigitalXRayImageStorageForPresentation
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(None,{},file_meta=meta,preamble=b'\0'*128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = 'DX'
    ds.BodyPartExamined = 'CHEST'
    ds.ViewPosition = 'PA'
    ds.PatientName = 'SYNTHETIC^NOT_A_PERSON'
    ds.PatientID = 'SYNTHETIC_ONLY'
    ds.Rows = ds.Columns = 64
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME1'
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.WindowCenter = 2048
    ds.WindowWidth = 4096
    ds.NumberOfFrames = frames
    ds.PixelData = np.arange(4096,dtype='<u2').reshape(64,64).tobytes()*frames
    output = io.BytesIO()
    dcmwrite(output,ds,enforce_file_format=True)
    return output.getvalue()


class ResearchCoreChecks(unittest.TestCase):
    def test_crop_projection_preserves_unassessed_regions(self):
        image = Image.new('RGB',(101,80))
        self.assertEqual(crop_box(image),(10,0,90,80))
        preview = preview_image(image)
        mask,box = project_map(np.ones((8,8)),image,preview,Image.Resampling.NEAREST)
        pixels = np.asarray(mask)
        self.assertEqual(box,(10,0,90,80))
        self.assertFalse(pixels[:,:10].any())
        self.assertFalse(pixels[:,90:].any())
        self.assertTrue((pixels[:,10:90] == 255).all())

    def test_quality_does_not_claim_clinical_acceptance(self):
        result = inspect_quality(Image.new('RGB',(128,128),(100,100,100)))
        self.assertEqual(result['status'],'review_suggested')
        self.assertFalse(result['validated_for_clinical_rejection'])

    def test_dicom_window_inversion_and_metadata_exclusion(self):
        image,metadata = decode_input(synthetic_dicom())
        pixels = np.asarray(image.convert('L'))
        self.assertEqual(image.size,(64,64))
        self.assertEqual(metadata['recorded_view'],'PA')
        self.assertGreater(pixels[0,0],pixels[-1,-1])
        self.assertFalse(metadata['patient_metadata_returned'])
        self.assertNotIn('SYNTHETIC',json.dumps(metadata))

    def test_multiframe_dicom_rejected(self):
        with self.assertRaisesRegex(ValueError,'single-frame'):
            decode_input(synthetic_dicom(frames=2))

    def test_aggregate_metrics_and_duplicate_rejection(self):
        rows = [{'case_id':f'case_{i}','label':'Pneumonia','truth':truth,'score':score}
                for i,(truth,score) in enumerate([(1,.9),(0,.1),(1,.8),(0,.2)])]
        payload = EvaluationRequest(model_weights='densenet121-res224-all',checkpoint_sha256='a'*64,observations=rows)
        report = evaluate_observations(payload,{'Pneumonia':.5})
        self.assertEqual(report['metrics']['Pneumonia']['auroc'],1)
        self.assertEqual(report['metrics']['Pneumonia']['tp'],2)
        self.assertEqual(report['metrics']['Pneumonia']['tn'],2)
        self.assertFalse(report['provenance_verified'])
        self.assertNotIn('case_0',json.dumps(report))
        payload.observations.append(payload.observations[0])
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            evaluate_observations(payload,{'Pneumonia':.5})

    def test_heatmap_uses_gradients_without_accumulating_parameter_grads(self):
        import torch
        from lunglens.classifier import Classifier
        from lunglens.config import Settings
        class SmallModel(torch.nn.Module):
            pathologies = ['Pneumonia']
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Conv2d(1,2,1,bias=False)
                self.classifier = torch.nn.Linear(2,1,bias=False)
                torch.nn.init.constant_(self.features.weight,.001)
                torch.nn.init.constant_(self.classifier.weight,.1)
        model = SmallModel().eval()
        classifier = Classifier(Settings())
        classifier.model = model
        values = np.tile(np.arange(256,dtype='uint8'),(320,1))
        result = classifier.heatmap(Image.fromarray(values).convert('RGB'),'PA','Pneumonia')
        self.assertEqual(result['finding'],'Pneumonia')
        self.assertEqual(result['status'],'generated')
        self.assertEqual(result['crop_box'],[0,32,256,288])
        overlay = Image.open(io.BytesIO(base64.b64decode(result['overlay_png_base64'])))
        self.assertEqual(overlay.size,(256,320))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == '__main__':
    unittest.main()
