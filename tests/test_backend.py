import io
import unittest
from unittest.mock import patch
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient
import backend
from lunglens.classifier import ModelUnavailable


def image_bytes():
    pixels = np.arange(128*128,dtype=np.uint8).reshape(128,128)
    out = io.BytesIO()
    Image.fromarray(pixels).save(out,format='PNG')
    return out.getvalue()


class FakeClassifier:
    def info(self):
        return {'loaded':True}
    def ready(self):
        return self.info()
    def predict(self,image,view):
        labels = ['Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema','Emphysema','Fibrosis','Effusion','Pneumonia','Pleural_Thickening','Cardiomegaly','Nodule','Mass','Hernia','Lung Lesion','Fracture','Lung Opacity','Enlarged Cardiomediastinum']
        return {'analysis_id':'test','created_at':'2026-10-27T00:00:00Z','project':'LungLens','task':'chest_xray_findings',
                'model':self.info(),'preprocessing':'test','threshold_policy':'test','notice':'test','limitations':[],
                'findings':[{'label':label,'score':.4,'reference_threshold':.2,'above_reference_threshold':True} for label in labels],
                'image':{'width':image.width,'height':image.height,'view':view},'score_type':'uncalibrated_sigmoid'}


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(backend.app)

    def test_health_and_model_readiness(self):
        with patch.object(backend,'classifier',FakeClassifier()):
            self.assertEqual(self.client.get('/health').status_code,200)
            self.assertTrue(self.client.get('/ready').json()['ready'])
            self.assertEqual(self.client.get('/models').status_code,200)

    def test_invalid_images_and_view(self):
        self.assertEqual(self.client.post('/predict?view=PA',content=b'bad').status_code,422)
        self.assertEqual(self.client.post('/predict?view=lateral',content=image_bytes()).status_code,422)
        out = io.BytesIO()
        Image.new('L',(128,128)).save(out,format='PNG')
        self.assertEqual(self.client.post('/predict?view=PA',content=out.getvalue()).status_code,422)
        out = io.BytesIO()
        Image.new('RGB',(128,128),(255,0,0)).save(out,format='PNG')
        self.assertEqual(self.client.post('/predict?view=PA',content=out.getvalue()).status_code,422)
        out = io.BytesIO()
        Image.new('I;16',(128,128)).save(out,format='PNG')
        self.assertEqual(self.client.post('/predict?view=PA',content=out.getvalue()).status_code,422)

    def test_view_is_optional(self):
        with patch.object(backend,'classifier',FakeClassifier()):
            response = self.client.post('/predict',content=image_bytes())
        self.assertEqual(response.status_code,200)
        self.assertIsNone(response.json()['image'].get('view'))

    def test_upload_limit(self):
        with patch.object(backend,'MAX_BYTES',4):
            self.assertEqual(self.client.post('/predict?view=PA',content=b'12345').status_code,413)

    def test_classifier_failure_returns_503(self):
        with patch.object(backend.classifier,'predict',side_effect=ModelUnavailable('Missing weights')):
            response = self.client.post('/predict?view=PA',content=image_bytes())
        self.assertEqual(response.status_code,503)
        self.assertEqual(response.json()['detail'],'Missing weights')

    def test_optional_explanation_does_not_change_scores(self):
        with patch.object(backend,'classifier',FakeClassifier()):
            a = self.client.post('/predict?view=PA',content=image_bytes()).json()
            with patch.object(backend.explainer,'explain',return_value={'status':'unavailable','detail':'offline'}):
                response = self.client.post('/predict?view=PA&explain=true',content=image_bytes())
        self.assertEqual(response.status_code,200)
        b = response.json()
        self.assertEqual(a['findings'],b['findings'])
        self.assertEqual(b['explanation']['status'],'unavailable')
        self.assertEqual(a['explanation']['status'],'not_requested')

    def test_png_and_jpeg_in_openapi(self):
        schema = self.client.get('/openapi.json').json()
        types = schema['paths']['/predict']['post']['requestBody']['content']
        self.assertIn('image/png',types)
        self.assertIn('image/jpeg',types)


if __name__ == '__main__':
    unittest.main()
