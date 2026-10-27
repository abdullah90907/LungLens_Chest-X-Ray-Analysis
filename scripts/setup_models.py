"""Download official model assets explicitly, outside request handling."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lunglens.config import Settings


def download_anatomy(settings):
    import requests
    from lunglens.anatomy import Anatomy, WEIGHTS_URL
    service = Anatomy(settings)
    target = service.cache_dir / WEIGHTS_URL.rsplit('/',1)[-1]
    manifest = service.cache_dir / 'checkpoint.json'
    if target.is_file() and manifest.is_file():
        record = json.loads(manifest.read_text())
        with target.open('rb') as checkpoint:
            digest = hashlib.file_digest(checkpoint,'sha256').hexdigest()
        if digest == record.get('sha256') and record.get('source') == WEIGHTS_URL:
            print(json.dumps(service.ready(),indent=2))
            return
    service.cache_dir.mkdir(parents=True,exist_ok=True)
    temporary = target.with_suffix('.pth.part')
    try:
        print('Downloading official ChestX-Det anatomical checkpoint...',flush=True)
        with requests.get(WEIGHTS_URL,stream=True,timeout=(15,120),headers={'Accept-Encoding':'identity'}) as response:
            response.raise_for_status()
            expected = int(response.headers.get('Content-Length',0))
            received = 0
            with temporary.open('wb') as output:
                for chunk in response.iter_content(1024*1024):
                    output.write(chunk)
                    received += len(chunk)
                    if received//(25*1024*1024) != (received-len(chunk))//(25*1024*1024):
                        print(f'Downloaded {received//(1024*1024)} MB',flush=True)
        if received < 1024*1024 or (expected and received != expected):
            raise RuntimeError('Incomplete anatomy checkpoint download')
        temporary.replace(target)
        with target.open('rb') as checkpoint:
            digest = hashlib.file_digest(checkpoint,'sha256').hexdigest()
        manifest.write_text(json.dumps({'source':WEIGHTS_URL,'sha256':digest},indent=2))
        print(json.dumps(service.ready(),indent=2))
    finally:
        temporary.unlink(missing_ok=True)


def download_classifier(settings):
    import requests
    import torchxrayvision as xrv
    from lunglens.classifier import Classifier
    url = xrv.models.model_urls[settings.weights]['weights_url']
    target = settings.cache_dir / url.rsplit('/', 1)[-1]
    manifest = settings.cache_dir / 'checkpoint.json'
    if target.is_file() and manifest.is_file():
        record = json.loads(manifest.read_text())
        with target.open('rb') as checkpoint:
            digest = hashlib.file_digest(checkpoint, 'sha256').hexdigest()
        if digest == record.get('sha256') and url == record.get('source'):
            print(json.dumps(Classifier(settings).ready(), indent=2))
            return
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.part')
    try:
        print('Downloading official TorchXRayVision checkpoint...', flush=True)
        with requests.get(url, stream=True, timeout=(15,120), headers={'Accept-Encoding':'identity'}) as response:
            response.raise_for_status()
            expected = int(response.headers.get('Content-Length', 0))
            received = 0
            with temporary.open('wb') as output:
                for chunk in response.iter_content(1024*1024):
                    output.write(chunk)
                    received += len(chunk)
                    if received // (5*1024*1024) != (received-len(chunk)) // (5*1024*1024):
                        print(f'Downloaded {received//(1024*1024)} MB', flush=True)
        if received < 1024*1024 or (expected and received != expected):
            raise RuntimeError('Checkpoint download is incomplete')
        temporary.replace(target)
        ready = Classifier(settings).ready()
        manifest.write_text(json.dumps({'source':url,'sha256':ready['checkpoint_sha256'],
                                        'weights':settings.weights}, indent=2))
        print(json.dumps(ready, indent=2))
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classifier', action='store_true')
    parser.add_argument('--medgemma', action='store_true')
    parser.add_argument('--anatomy', action='store_true')
    args = parser.parse_args()
    if not (args.classifier or args.medgemma or args.anatomy):
        parser.error('Choose --classifier, --anatomy and/or --medgemma')
    settings = Settings.from_env()
    if args.classifier:
        download_classifier(settings)
    if args.anatomy:
        download_anatomy(settings)
    if args.medgemma:
        from huggingface_hub import snapshot_download, HfApi, get_token
        from huggingface_hub.errors import GatedRepoError
        if not get_token():
            parser.exit(2, 'Hugging Face login required. Accept the chosen MedGemma model terms, then run .venv\\Scripts\\hf.exe auth login locally. Do not paste a token into chat.\n')
        try:
            info = HfApi().model_info(settings.medgemma_model, revision=settings.medgemma_revision)
            path = snapshot_download(settings.medgemma_model, revision=info.sha,
                                     allow_patterns=['*.json','*.safetensors','*.model','*.jinja','*.txt'])
            print(json.dumps({'downloaded':True,'model':settings.medgemma_model,'revision':info.sha,'cache':path}))
            print('Pin LUNGLENS_MEDGEMMA_REVISION to this revision in .env before use.')
        except GatedRepoError:
            parser.exit(2, 'MedGemma access required: accept terms on its Hugging Face page, then run .venv\\Scripts\\hf.exe auth login locally. Never paste your token into chat.\n')


if __name__ == '__main__':
    main()
