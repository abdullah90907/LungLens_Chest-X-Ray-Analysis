import base64
import io
import numpy as np
from PIL import Image


def crop_box(image):
    side = min(image.size)
    x = image.width // 2 - side // 2
    y = image.height // 2 - side // 2
    return (x, y, x + side, y + side)


def png_base64(image):
    output = io.BytesIO()
    image.save(output, format='PNG')
    return base64.b64encode(output.getvalue()).decode('ascii')


def preview_image(image, max_side=1024):
    preview = image.copy()
    preview.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return preview


def project_map(values, image, preview, resample=Image.Resampling.BILINEAR):
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Invalid visualization array')
    box = crop_box(image)
    sx, sy = preview.width / image.width, preview.height / image.height
    projected_box = (round(box[0]*sx), round(box[1]*sy), round(box[2]*sx), round(box[3]*sy))
    patch = Image.fromarray(np.clip(values*255, 0, 255).astype('uint8')).resize(
        (projected_box[2]-projected_box[0], projected_box[3]-projected_box[1]), resample)
    canvas = Image.new('L', preview.size, 0)
    canvas.paste(patch, projected_box[:2])
    return canvas, projected_box


def heatmap_visuals(values, image):
    preview = preview_image(image)
    gray, box = project_map(values, image, preview)
    intensity = np.asarray(gray, dtype=np.float32) / 255
    stops = np.array([[36,110,187], [25,174,151], [229,210,91], [240,123,51]], dtype=np.float32)
    position = intensity * 3
    lower = np.minimum(position.astype(int), 2)
    fraction = (position-lower)[..., None]
    rgb = stops[lower]*(1-fraction) + stops[lower+1]*fraction
    rgba = np.concatenate((rgb, (intensity*175)[..., None]), axis=2).astype('uint8')
    color = Image.fromarray(rgba)
    combined = Image.alpha_composite(preview.convert('RGBA'), color).convert('RGB')
    return {'width':preview.width, 'height':preview.height, 'crop_box':list(crop_box(image)),
            'preview_crop_box':list(box), 'heatmap_png_base64':png_base64(color),
            'overlay_png_base64':png_base64(combined),
            'coordinates':'Original-image crop_box; overlay and heatmap pixels use preview dimensions. Regions outside the crop are transparent.'}
