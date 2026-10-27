"""Offline synthetic pre-deployment checks. No files are uploaded or images saved."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from lunglens.config import Settings
from lunglens.classifier import Classifier
from lunglens.anatomy import Anatomy


def main():
    torch.set_num_threads(4)
    pixels = np.random.default_rng(21).integers(0,256,(320,256),dtype=np.uint8)
    image = Image.fromarray(pixels).convert('RGB')
    settings = Settings.from_env()
    classifier = Classifier(settings)
    print('Checking real-weight Grad-CAM in memory...',flush=True)
    heatmap = classifier.heatmap(image,'PA','Pneumonia')
    assert heatmap['width'] == 256 and heatmap['height'] == 320
    assert heatmap['crop_box'] == [0,32,256,288]
    assert heatmap['heatmap_png_base64'] and heatmap['overlay_png_base64']
    print('Checking real-weight anatomical segmentation in memory...',flush=True)
    result = Anatomy(settings).segment(image)
    assert len(result['structures']) == 3
    assert all(0 <= item['crop_area_fraction'] <= 1 for item in result['structures'])
    assert result['overlay_png_base64']
    print('Checking API schema without HTTP requests...',flush=True)
    from backend import app
    schema = app.openapi()
    assert all(route in schema['paths'] for route in ['/research/inspect','/research/heatmap','/research/anatomy','/research/evaluate'])
    print(json.dumps({'preflight':'passed','heatmap_status':heatmap['status'],
                      'anatomy_structures':[item['label'] for item in result['structures']],
                      'notice':'Synthetic functionality checks only; no clinical accuracy evaluation'},indent=2))


if __name__ == '__main__':
    main()
