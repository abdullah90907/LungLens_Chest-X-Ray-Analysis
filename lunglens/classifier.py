import hashlib
import importlib.metadata
import json
import threading
from datetime import datetime, timezone
import uuid
import numpy as np
from .imaging import preprocess

NOTICE = 'Research prototype: model scores are not calibrated diagnostic probabilities. Low scores do not establish a normal X-ray. Mass/nodule findings do not confirm cancer.'


class ModelUnavailable(RuntimeError):
    pass


class Classifier:
    def __init__(self, settings):
        self.settings = settings
        self.lock = threading.Lock()
        self.model = None
        self.thresholds = None
        self.sha256 = None
        self.last_error = None

    def _load(self):
        if self.model is not None:
            return
        try:
            import torch
            import torchxrayvision as xrv
            if self.settings.device == 'cuda' and not torch.cuda.is_available():
                raise ModelUnavailable('CUDA requested but unavailable')
            meta = xrv.models.model_urls[self.settings.weights]
            filename = meta['weights_url'].rsplit('/', 1)[-1]
            path = self.settings.cache_dir / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise ModelUnavailable('Run python scripts/setup_models.py --classifier to download the pretrained weights')
            with path.open('rb') as checkpoint:
                digest = hashlib.file_digest(checkpoint, 'sha256').hexdigest()
            manifest = self.settings.cache_dir / 'checkpoint.json'
            if manifest.is_file():
                record = json.loads(manifest.read_text())
                if record.get('sha256') != digest or record.get('source') != meta['weights_url'] or record.get('weights') != self.settings.weights:
                    raise ModelUnavailable('Checkpoint integrity check failed; rerun scripts/setup_models.py --classifier')
            model = xrv.models.DenseNet(weights=self.settings.weights, cache_dir=str(self.settings.cache_dir))
            self.thresholds = model.op_threshs.detach().cpu().numpy().copy()
            # Use raw sigmoid scores, not XRV operating-point-normalized scores.
            model.op_threshs = None
            model.apply_sigmoid = True
            model.eval().to(self.settings.device)
            if (len(model.pathologies) != 18 or len(set(model.pathologies)) != 18
                    or self.thresholds.shape != (18,) or not np.isfinite(self.thresholds).all()
                    or np.any(self.thresholds <= 0) or np.any(self.thresholds >= 1)):
                raise ModelUnavailable('Unexpected classifier labels or thresholds')
            self.sha256 = digest
            self.model = model
            self.last_error = None
        except ModelUnavailable as exc:
            self.last_error = str(exc)
            raise
        except Exception as exc:
            self.last_error = 'Classifier failed to load; check installation and cached weights'
            raise ModelUnavailable(self.last_error) from exc

    def ready(self):
        with self.lock:
            self._load()
        return self.info()

    def info(self):
        return {'name': 'TorchXRayVision DenseNet121', 'weights': self.settings.weights,
                'loaded': self.model is not None, 'device': self.settings.device,
                'checkpoint_sha256': self.sha256, 'last_error': self.last_error}

    def predict(self, image, view):
        import torch
        import torchxrayvision as xrv
        with self.lock:
            self._load()
            array = preprocess(image, xrv)
            with torch.inference_mode():
                scores = self.model(torch.from_numpy(array[None]).to(self.settings.device)).detach().cpu().numpy()[0]
            if scores.shape != (18,) or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
                raise ModelUnavailable('Classifier returned invalid scores')
            findings = [
                {'label': label, 'score': float(score), 'reference_threshold': float(threshold),
                 'above_reference_threshold': bool(score >= threshold)}
                for label, score, threshold in zip(self.model.pathologies, scores, self.thresholds)
            ]
            findings.sort(key=lambda item: item['score'], reverse=True)
            return {
                'analysis_id': str(uuid.uuid4()), 'created_at': datetime.now(timezone.utc).isoformat(),
                'project': 'LungLens', 'task': 'chest_xray_findings',
                'model': {**self.info(), 'library_version': importlib.metadata.version('torchxrayvision')},
                'image': {'width': image.width, 'height': image.height, 'view': view,
                          'modality_verified': False},
                'preprocessing': 'RGB mean channel; [-1024,1024]; center crop; 224x224',
                'score_type': 'uncalibrated_sigmoid',
                'threshold_policy': 'Published XRV reference operating points; not locally validated clinical cutoffs',
                'findings': findings, 'notice': NOTICE,
                'limitations': ['Accept frontal AP/PA chest X-rays only; modality/view is declared by the user, not verified by a model.',
                                'Center cropping can remove peripheral anatomy.',
                                'No patient-level external evaluation has been performed for this application.'] + (['Image view was not provided; frontal AP/PA suitability has not been confirmed.'] if view is None else []),
            }

    def heatmap(self, image, view, finding):
        import torch
        import torch.nn.functional as F
        import torchxrayvision as xrv
        from .visuals import heatmap_visuals
        with self.lock:
            self._load()
            if finding not in self.model.pathologies:
                raise ValueError('Choose an exact finding name from /research/capabilities')
            index = self.model.pathologies.index(finding)
            tensor = torch.from_numpy(preprocess(image,xrv)[None]).to(self.settings.device).requires_grad_(True)
            with torch.enable_grad():
                features = F.relu(self.model.features(tensor), inplace=False)
                logits = self.model.classifier(F.adaptive_avg_pool2d(features,(1,1)).flatten(1))
                gradient = torch.autograd.grad(logits[0,index], features)[0]
                weighted = (gradient.mean(dim=(2,3),keepdim=True)*features).sum(dim=1,keepdim=True)
                cam = F.relu(weighted)
                cam = F.interpolate(cam,size=(224,224),mode='bilinear',align_corners=False)[0,0]
                peak = float(cam.max().detach().cpu())
                values = (cam/peak if peak > 0 else torch.zeros_like(cam)).detach().cpu().numpy()
                score = float(torch.sigmoid(logits[0,index]).detach().cpu())
            if not np.isfinite(values).all() or not np.isfinite(score):
                raise ModelUnavailable('Heatmap contains invalid attribution values')
            return {'method':'Grad-CAM', 'finding':finding, 'score':score, 'view':view,
                    'score_type':'uncalibrated_sigmoid',
                    'status':'generated' if peak > 0 else 'no_positive_attribution',
                    'target':'pre-sigmoid finding logit; final post-ReLU DenseNet feature map',
                    'model':self.info(), **heatmap_visuals(values,image),
                    'notice':'Relative model attribution, not disease segmentation or proof of a finding. Color intensity is normalized within this image and cannot be compared as severity across images.'}
