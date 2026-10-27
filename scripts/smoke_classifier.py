"""Verify real pretrained inference on synthetic pixels, without accuracy claims."""
import io
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from backend import app


def main():
    image = np.random.default_rng(42).integers(0,256,(320,256),dtype=np.uint8)
    data = io.BytesIO()
    Image.fromarray(image).save(data,format='PNG')
    with TestClient(app) as client:
        ready = client.get('/ready')
        if ready.status_code != 200:
            raise RuntimeError(ready.text)
        response = client.post('/predict?view=PA&explain=true',content=data.getvalue(),headers={'Content-Type':'image/png'})
        if response.status_code != 200:
            raise RuntimeError(response.text)
        result = response.json()
        assert len(result['findings']) == 18
        assert len({item['label'] for item in result['findings']}) == 18
        assert all(0 <= item['score'] <= 1 for item in result['findings'])
        assert result['score_type'] == 'uncalibrated_sigmoid'
        assert result['model']['checkpoint_sha256']
        assert not result['image']['modality_verified']
        print(json.dumps({'passed':True,'findings':len(result['findings']),
                          'model':result['model'],'explanation':result['explanation'],
                          'notice':'Synthetic smoke check only; not a medical performance evaluation'},indent=2))


if __name__ == '__main__':
    main()
