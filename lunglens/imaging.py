import io
import warnings
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 16_000_000
MIN_SIDE = 64


def decode_input(data: bytes, content_type=''):
    if not data:
        raise ValueError('Upload a PNG, JPEG or DICOM chest X-ray')
    if len(data) > MAX_BYTES:
        raise ValueError('Image exceeds 20 MB')
    if data[128:132] == b'DICM' or content_type.split(';')[0].lower() == 'application/dicom':
        from .dicom import decode_dicom
        image, metadata = decode_dicom(data, MAX_PIXELS, MIN_SIDE)
        if np.asarray(image.convert('L')).std() < 1:
            raise ValueError('Image is blank or has insufficient pixel variation')
        return image, metadata
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {'PNG', 'JPEG'}:
                    raise ValueError('Use an 8-bit PNG/JPEG export or a supported CR/DX DICOM file; TIFF is not supported')
                if source.width * source.height > MAX_PIXELS or min(source.size) < MIN_SIDE:
                    raise ValueError('Image must be at least 64 pixels per side and no more than 16 megapixels')
                if getattr(source, 'n_frames', 1) != 1:
                    raise ValueError('Upload a single image')
                if source.mode in {'I', 'F'} or source.mode.startswith('I;16'):
                    raise ValueError('Export a correctly windowed 8-bit PNG/JPEG first')
                if 'A' in source.getbands() or 'transparency' in source.info:
                    raise ValueError('Transparent images are not supported')
                image = ImageOps.exif_transpose(source).convert('RGB')
                # Variation across color channels alone does not make a real image.
                if np.asarray(image.convert('L')).std() < 1:
                    raise ValueError('Image is blank or has insufficient pixel variation')
                return image, {'format':source.format, 'recorded_view':None, 'windowing':'source_export'}
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('Invalid, corrupted, or oversized image') from exc


def decode_image(data: bytes) -> Image.Image:
    return decode_input(data)[0]


def preprocess(image, xrv, size=224):
    # Match the library example: normalize RGB, then average channels and crop/resize.
    pixels = np.asarray(image, dtype=np.float32)
    normalized = xrv.datasets.normalize(pixels, 255).mean(axis=2)[None, ...]
    cropped = xrv.datasets.XRayCenterCrop()(normalized)
    resized = xrv.datasets.XRayResizer(size)(cropped)
    return np.ascontiguousarray(resized, dtype=np.float32)
