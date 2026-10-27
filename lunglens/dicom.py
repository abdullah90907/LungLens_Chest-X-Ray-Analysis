import io
import warnings
import numpy as np
from PIL import Image


class DicomInputError(ValueError):
    pass


def decode_dicom(data, max_pixels, min_side):
    try:
        import pydicom
        from pydicom.pixels import apply_modality_lut, apply_voi_lut
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            ds = pydicom.dcmread(io.BytesIO(data), force=False)
            rows, columns = int(ds.Rows), int(ds.Columns)
            if min(rows, columns) < min_side or rows*columns > max_pixels:
                raise DicomInputError('DICOM dimensions exceed supported image limits')
            if int(getattr(ds,'NumberOfFrames',1)) != 1 or int(ds.SamplesPerPixel) != 1:
                raise DicomInputError('Only single-frame monochrome chest radiographs are supported')
            if str(getattr(ds,'Modality','')).upper() not in {'CR','DX'}:
                raise DicomInputError('Only CR/DX radiographs are supported; CT and other modalities are not accepted')
            body_part = str(getattr(ds,'BodyPartExamined','')).strip().upper()
            if body_part and body_part not in {'CHEST','THORAX'}:
                raise DicomInputError('DICOM anatomy metadata does not identify a chest radiograph')
            interpretation = str(ds.PhotometricInterpretation)
            if interpretation not in {'MONOCHROME1','MONOCHROME2'}:
                raise DicomInputError('Only MONOCHROME1/2 DICOM images are supported')
            if int(ds.BitsAllocated) not in {8,16} or not 1 <= int(ds.BitsStored) <= int(ds.BitsAllocated):
                raise DicomInputError('Only 8/16-bit integer DICOM pixels are supported')
            if int(ds.PixelRepresentation) not in {0,1}:
                raise DicomInputError('Unsupported DICOM pixel representation')
            if 'PresentationLUTSequence' in ds:
                raise DicomInputError('Presentation LUT sequences require a properly rendered PNG/JPEG export')
            pixels = ds.pixel_array
            if pixels.shape != (rows, columns):
                raise DicomInputError('Unexpected DICOM pixel dimensions')
            padding = np.zeros(pixels.shape, dtype=bool)
            if 'PixelPaddingValue' in ds:
                low = float(ds.PixelPaddingValue)
                high = float(getattr(ds,'PixelPaddingRangeLimit',low))
                padding = (pixels >= min(low,high)) & (pixels <= max(low,high))
            modality_pixels = apply_modality_lut(pixels, ds)
            mapped = np.asarray(apply_voi_lut(modality_pixels, ds, index=0),dtype=np.float64)
            if not np.isfinite(mapped).all() or padding.all():
                raise DicomInputError('DICOM contains invalid or empty pixel data')
            if 'VOILUTSequence' in ds:
                low, high = 0, 2**int(ds.VOILUTSequence[0].LUTDescriptor[2])-1
                window = 'dicom_voi_lut_first'
            elif 'WindowCenter' in ds and 'WindowWidth' in ds:
                if 'ModalityLUTSequence' in ds:
                    low, high = 0, 2**int(ds.ModalityLUTSequence[0].LUTDescriptor[2])-1
                else:
                    bits, signed = int(ds.BitsStored), int(ds.PixelRepresentation)
                    bounds = np.array([-(2**(bits-1)),2**(bits-1)-1] if signed else [0,2**bits-1])
                    bounds = np.asarray(apply_modality_lut(bounds,ds),dtype=float)
                    low, high = float(bounds.min()), float(bounds.max())
                window = 'dicom_window_first'
            else:
                low, high = np.percentile(mapped[~padding],[.5,99.5])
                window = 'automatic_percentile_0.5_99.5'
            if high <= low:
                raise DicomInputError('DICOM has no usable intensity range')
            gray = np.clip((mapped-low)/(high-low),0,1)
            shape = str(getattr(ds,'PresentationLUTShape','IDENTITY')).upper()
            if shape not in {'IDENTITY','INVERSE'}:
                raise DicomInputError('Unsupported DICOM presentation LUT shape')
            if interpretation == 'MONOCHROME1' or shape == 'INVERSE':
                gray = 1-gray
            gray[padding] = 0
            image = Image.fromarray(np.round(gray*255).astype('uint8')).convert('RGB')
            view = str(getattr(ds,'ViewPosition','')).strip().upper()
            if view in {'LL','RL','LATERAL','LATERAL LEFT','LATERAL RIGHT'}:
                raise DicomInputError('Lateral DICOM views are not supported by the frontal classifier')
            info = {'format':'DICOM', 'dicom_modality':str(ds.Modality),
                    'recorded_view':view if view in {'AP','PA'} else None,
                    'windowing':window, 'patient_metadata_returned':False,
                    'notice':'Only whitelisted imaging metadata is returned. Pixel data may contain burned-in identifiers; this is not certified de-identification.'}
            return image, info
    except DicomInputError:
        raise
    except Exception:
        raise DicomInputError('DICOM could not be decoded. Use a valid single-frame CR/DX file with supported compression, or export an 8-bit PNG/JPEG.') from None
