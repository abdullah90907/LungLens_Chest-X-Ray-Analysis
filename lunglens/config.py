"""Environment configuration; all model caches default to this project."""
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env', override=False)
# Keep standard HF login-token discovery; relocate only the model download cache.
os.environ.setdefault('HF_HUB_CACHE', str(ROOT / 'models' / 'huggingface' / 'hub'))


@dataclass(frozen=True)
class Settings:
    weights: str = 'densenet121-res224-all'
    device: str = 'cpu'
    cache_dir: Path = ROOT / 'models' / 'torchxrayvision'
    medgemma_mode: str = 'disabled'
    medgemma_model: str = 'google/medgemma-1.5-4b-it'
    medgemma_revision: str = 'main'
    medgemma_url: str = ''
    medgemma_api_key: str = ''
    medgemma_allow_cpu: bool = False
    allowed_origins: tuple = ()

    @classmethod
    def from_env(cls):
        mode = os.getenv('LUNGLENS_MEDGEMMA_MODE', 'disabled')
        if mode not in {'disabled', 'local', 'remote'}:
            raise ValueError('LUNGLENS_MEDGEMMA_MODE must be disabled, local or remote')
        device = os.getenv('LUNGLENS_DEVICE', 'cpu')
        if device not in {'cpu', 'cuda'}:
            raise ValueError('LUNGLENS_DEVICE must be cpu or cuda')
        return cls(
            device=device,
            cache_dir=Path(os.getenv('LUNGLENS_CACHE_DIR', str(cls.cache_dir))),
            medgemma_mode=mode,
            medgemma_model=os.getenv('LUNGLENS_MEDGEMMA_MODEL', cls.medgemma_model),
            medgemma_revision=os.getenv('LUNGLENS_MEDGEMMA_REVISION', 'main'),
            medgemma_url=os.getenv('LUNGLENS_MEDGEMMA_URL', '').rstrip('/'),
            medgemma_api_key=os.getenv('LUNGLENS_MEDGEMMA_API_KEY', ''),
            medgemma_allow_cpu=os.getenv('LUNGLENS_MEDGEMMA_ALLOW_CPU', 'false').lower() == 'true',
            allowed_origins=tuple(x.strip() for x in os.getenv('LUNGLENS_ALLOWED_ORIGINS', '').split(',') if x.strip()),
        )
