import numpy as np
from PIL import Image


def inspect_quality(image):
    sample = image.convert('L')
    sample.thumbnail((512,512), Image.Resampling.BILINEAR)
    pixels = np.asarray(sample, dtype=np.float32)
    p1, p99 = np.percentile(pixels, [1,99])
    laplacian = pixels[1:-1,:-2] + pixels[1:-1,2:] + pixels[:-2,1:-1] + pixels[2:,1:-1] - 4*pixels[1:-1,1:-1]
    sharpness = float(laplacian.var())
    dark, bright = float((pixels <= 2).mean()), float((pixels >= 253).mean())
    flags = []
    if p99-p1 < 40:
        flags.append({'code':'low_contrast', 'message':'The image has a narrow intensity range. Check its source windowing and export.'})
    if sharpness < 8:
        flags.append({'code':'low_local_detail', 'message':'Low local image detail was measured. This can reflect blur, resizing or the image content.'})
    if dark > .25 or bright > .25:
        flags.append({'code':'extreme_pixel_fraction', 'message':'Many pixels are near black or white. Check for clipping; borders and backgrounds may also cause this.'})
    if not .65 <= image.width/image.height <= 1.3:
        flags.append({'code':'unusual_aspect_ratio', 'message':'Check the image orientation and field of view. The classifier uses a center crop.'})
    return {'status':'review_suggested' if flags else 'no_heuristic_flags',
            'flags':flags, 'measurements':{'intensity_p01':float(p1), 'intensity_p99':float(p99),
                'laplacian_variance_512':sharpness, 'near_black_fraction':dark, 'near_white_fraction':bright},
            'validated_for_clinical_rejection':False,
            'notice':'Heuristic research flags only. No flags does not establish a suitable image, correct modality, or diagnostic quality.'}
