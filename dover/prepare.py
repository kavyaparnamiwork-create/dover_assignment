"""Step 1: load, orient, grayscale, contrast-stretch. Deterministic, same for every image."""
import cv2
import numpy as np
from PIL import Image, ImageOps

try:  # optional: lets the same code read the original .heic photos
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

WORK_LONG_SIDE = 2400


def load_image(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def enhance(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def prepare(bgr):
    """Returns (full_res_enhanced_gray, working_gray, scale). Boxes live in working coords;
    crops for the second read are cut from the full-res image."""
    gray = enhance(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    scale = WORK_LONG_SIDE / max(gray.shape)
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    work = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)
    return gray, work, scale
