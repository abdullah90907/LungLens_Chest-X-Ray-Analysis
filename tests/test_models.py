import io
import json
import unittest
import tempfile
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from PIL import Image
from lunglens.config import Settings
from lunglens.explainer import Explainer
from lunglens.imaging import preprocess, decode_image


class ModelTests(unittest.TestCase):
    def test_corrupt_checkpoint_rejected_before_model_load(self):
        import torchxrayvision as xrv
        from lunglens.classifier import Classifier, ModelUnavailable
        settings = Settings()
        url = xrv.models.model_urls[settings.weights]['weights_url']
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            (root/url.rsplit('/',1)[-1]).write_bytes(b'corrupted checkpoint')
            (root/'checkpoint.json').write_text(json.dumps({'sha256':'incorrect','source':url,'weights':settings.weights}))
            with patch.object(xrv.models,'DenseNet') as loader:
                with self.assertRaisesRegex(ModelUnavailable,'integrity check failed'):
                    Classifier(replace(settings,cache_dir=root)).ready()
                loader.assert_not_called()

    def test_preprocessing_matches_official_example(self):
        import torchxrayvision as xrv
        image = Image.fromarray(np.random.default_rng(1).integers(0,256,(256,320,3),dtype=np.uint8))
        expected = xrv.datasets.normalize(np.asarray(image),255).mean(2)[None,...]
        expected = xrv.datasets.XRayResizer(224)(xrv.datasets.XRayCenterCrop()(expected))
        actual = preprocess(image,xrv)
        np.testing.assert_allclose(actual,expected,atol=.001)
        self.assertEqual(actual.shape,(1,224,224))
        self.assertEqual(actual.dtype,np.float32)
        self.assertTrue(actual.flags.c_contiguous)

    def test_classifier_independent_sigmoid_outputs(self):
        import torch
        import torchxrayvision as xrv
        from lunglens.classifier import Classifier
        class Fake:
            pathologies = xrv.datasets.default_pathologies
            def __call__(self, tensor):
                self.tensor = tensor
                return torch.full((1,18),.8)
        classifier = Classifier(Settings())
        classifier.model = Fake()
        classifier.thresholds = np.full(18,.2)
        result = classifier.predict(Image.new('RGB',(256,320)), 'AP')
        self.assertEqual(len(result['findings']),18)
        self.assertGreater(sum(item['score'] for item in result['findings']),1)
        self.assertTrue(all(item['above_reference_threshold'] for item in result['findings']))
        self.assertFalse(result['image']['modality_verified'])

    def test_explanation_only_accepts_supplied_labels(self):
        findings = [{'label':'Pneumonia','score':.3,'above_reference_threshold':True}]
        valid = json.dumps({'items':[{'label':'Pneumonia','explanation':'A term for infection affecting lung tissue.'}]})
        self.assertEqual(Explainer.parse(valid,findings)[0]['label'],'Pneumonia')
        with self.assertRaises(ValueError):
            Explainer.parse(valid.replace('Pneumonia','Cancer'),findings)
        with self.assertRaises(ValueError):
            Explainer.parse('{"items":[]}',findings)
        self.assertIn('You have not seen the X-ray',Explainer.prompt(findings))

    def test_disabled_and_unavailable_explanations(self):
        result = {'findings':[{'label':'Mass','score':.3,'above_reference_threshold':True}]}
        self.assertEqual(Explainer(Settings()).explain(result)['status'],'disabled')
        configured = Explainer(replace(Settings(),medgemma_mode='remote'))
        self.assertEqual(configured.explain(result)['status'],'unavailable')

    def test_remote_explanation_path(self):
        import httpx
        findings = [{'label':'Effusion','score':.4,'above_reference_threshold':True}]
        payload = {'choices':[{'message':{'content':json.dumps({'items':[{'label':'Effusion','explanation':'Fluid collecting around a lung.'}]})}}]}
        response = httpx.Response(200,json=payload,request=httpx.Request('POST','https://example.test/v1/chat/completions'))
        service = Explainer(replace(Settings(),medgemma_mode='remote',medgemma_url='https://example.test/v1'))
        with patch('lunglens.explainer.httpx.Client') as client:
            client.return_value.__enter__.return_value.post.return_value = response
            result = service.explain({'findings':findings})
            self.assertEqual(result['status'],'generated')
            self.assertTrue(result['review_required'])
            sent = client.return_value.__enter__.return_value.post.call_args.kwargs['json']
            self.assertEqual(sent['temperature'],0)
        self.assertEqual(findings[0]['score'],.4)


if __name__ == '__main__':
    unittest.main()
