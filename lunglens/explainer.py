"""Optional MedGemma explanations of fixed classifier findings, without image re-reading."""
import json
import threading
import httpx


class ExplanationUnavailable(RuntimeError):
    pass


class Explainer:
    def __init__(self, settings):
        self.settings = settings
        self.model = self.processor = None
        self.lock = threading.Lock()
        self.last_error = None

    def info(self):
        return {'mode': self.settings.medgemma_mode, 'model': self.settings.medgemma_model,
                'revision': self.settings.medgemma_revision,
                'loaded': self.model is not None, 'last_error': self.last_error,
                'role': 'Explain supplied classifier findings; no independent image interpretation'}

    def _load(self):
        if self.settings.medgemma_mode == 'disabled':
            raise ExplanationUnavailable('MedGemma is disabled; enable local or remote mode in .env')
        if self.settings.medgemma_mode == 'remote':
            if not self.settings.medgemma_url:
                raise ExplanationUnavailable('Configure a MedGemma OpenAI-compatible /v1 endpoint')
            return
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
            if not torch.cuda.is_available() and not self.settings.medgemma_allow_cpu:
                raise ExplanationUnavailable('Local MedGemma needs CUDA by default; configure a hosted endpoint or explicitly allow slow CPU inference')
            options = {'revision': self.settings.medgemma_revision, 'local_files_only': True,
                       'trust_remote_code': False}
            processor = AutoProcessor.from_pretrained(self.settings.medgemma_model, **options)
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)
            model = AutoModelForImageTextToText.from_pretrained(
                self.settings.medgemma_model, torch_dtype=dtype,
                device_map='auto' if torch.cuda.is_available() else 'cpu', **options)
            self.processor, self.model = processor, model.eval()
        except ExplanationUnavailable:
            raise
        except Exception as exc:
            raise ExplanationUnavailable('MedGemma could not load. Accept its Hugging Face terms, authenticate locally and run scripts/setup_models.py --medgemma') from exc

    def ready(self):
        try:
            with self.lock:
                self._load()
                if self.settings.medgemma_mode == 'remote':
                    headers = self._headers()
                    with httpx.Client(timeout=15) as client:
                        response = client.get(self.settings.medgemma_url + '/models', headers=headers)
                        response.raise_for_status()
                        ids = {item['id'] for item in response.json()['data']}
                        if self.settings.medgemma_model not in ids:
                            raise ExplanationUnavailable('Configured MedGemma model is not listed by the endpoint')
            self.last_error = None
            return {**self.info(), 'ready': True}
        except ExplanationUnavailable as exc:
            self.last_error = str(exc)
            raise
        except Exception as exc:
            self.last_error = 'MedGemma endpoint is unavailable or incompatible'
            raise ExplanationUnavailable(self.last_error) from exc

    def _headers(self):
        return {'Authorization': 'Bearer ' + self.settings.medgemma_api_key} if self.settings.medgemma_api_key else {}

    @staticmethod
    def prompt(findings):
        context = [{'label': row['label'], 'score': row['score'],
                    'above_reference_threshold': row['above_reference_threshold']} for row in findings]
        return (
            'Explain the supplied research classifier findings in plain language. You have not seen the X-ray. '
            'Scores are uncalibrated and reference thresholds are not clinically validated for this user. '
            'Do not diagnose, infer severity, assert absence of disease, recommend treatments, or add findings. '
            'For each supplied label, describe only what that term generally means; do not claim it is present. '
            'Return only JSON in this form: {"items":[{"label":"exact supplied label","explanation":"one short informational sentence"}]}. '
            'Include each supplied label exactly once. Data: ' + json.dumps(context)
        )

    @staticmethod
    def parse(text, findings):
        text = text.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        payload = json.loads(text)
        items = payload['items']
        expected = {row['label'] for row in findings}
        if not isinstance(items, list) or len(items) != len(expected):
            raise ValueError('Explanation labels do not match classifier context')
        labels = [item['label'] for item in items]
        if set(labels) != expected or len(set(labels)) != len(labels):
            raise ValueError('Explanation introduced or omitted labels')
        result = []
        for item in items:
            explanation = item['explanation']
            if not isinstance(explanation, str) or not 1 <= len(explanation) <= 1200:
                raise ValueError('Invalid explanation length')
            result.append({'label': item['label'], 'explanation': explanation})
        return result

    def explain(self, result):
        if self.settings.medgemma_mode == 'disabled':
            return {'status': 'disabled', 'detail': 'MedGemma is not enabled'}
        selected = [row for row in result['findings'] if row['above_reference_threshold']][:5]
        if not selected:
            return {'status': 'not_generated', 'detail': 'No finding exceeded a reference threshold. This does not establish a normal image.'}
        try:
            with self.lock:
                self._load()
                prompt = self.prompt(selected)
                if self.settings.medgemma_mode == 'remote':
                    with httpx.Client(timeout=httpx.Timeout(90, connect=10)) as client:
                        response = client.post(self.settings.medgemma_url + '/chat/completions', headers=self._headers(), json={
                            'model': self.settings.medgemma_model,
                            'messages': [{'role': 'user', 'content': prompt}],
                            'temperature': 0, 'max_tokens': 700,
                        })
                        response.raise_for_status()
                        text = response.json()['choices'][0]['message']['content']
                else:
                    import torch
                    inputs = self.processor.apply_chat_template(
                        [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}],
                        add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to(self.model.device)
                    with torch.inference_mode():
                        output = self.model.generate(**inputs, max_new_tokens=700, do_sample=False, max_time=90)
                    text = self.processor.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
                items = self.parse(text, selected)
                self.last_error = None
                return {'status': 'generated', 'model': self.settings.medgemma_model,
                        'items': items, 'review_required': True,
                        'notice': 'AI-generated educational explanations may contain errors. They do not change classifier scores and are not an independent diagnosis.'}
        except Exception as exc:
            self.last_error = str(exc) if isinstance(exc, ExplanationUnavailable) else 'Explanation unavailable or generated output failed validation'
            return {'status': 'unavailable', 'detail': self.last_error}
