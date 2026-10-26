"""AI part 1 - OCR: read the board label from a photo (Tesseract)."""
import os

# One thread per Tesseract call: its OpenMP threading slows small images down
# badly, especially when several photos are read in parallel.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import difflib
import re

import numpy as np
import pytesseract
from PIL import Image, ImageOps
from scipy import ndimage


def preprocess(img: Image.Image) -> Image.Image:
    """Grayscale -> upscale -> contrast stretch -> Otsu binarisation."""
    g = ImageOps.grayscale(img)
    if g.width < 1600:
        s = 1600 / g.width
        g = g.resize((int(g.width * s), int(g.height * s)), Image.LANCZOS)
    g = ImageOps.autocontrast(g, cutoff=2)
    arr = np.asarray(g)
    # Otsu threshold
    hist = np.bincount(arr.ravel(), minlength=256).astype(float)
    total, sum_all = arr.size, np.dot(np.arange(256), hist)
    w_b = sum_b = 0.0
    best_t, best_var = 128, 0.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0 or w_b == total:
            continue
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / (total - w_b)
        var = w_b * (total - w_b) * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return Image.fromarray(((arr > best_t) * 255).astype(np.uint8))


def _best_region(mask, min_frac=0.004, min_fill=0.6, aspect=None):
    """Largest connected region that is close to a filled rectangle (a tag).
    aspect=(lo, hi) restricts width/height (either orientation)."""
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=4))
    lab, n = ndimage.label(mask)
    best, best_area = None, min_frac * mask.size
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        area = (lab[sl] == i).sum()
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        ratio = max(w, h) / max(1, min(w, h))
        if aspect and not (aspect[0] <= ratio <= aspect[1]):
            continue    # e.g. long gold edge-connector fingers are yellow too
        if area > best_area and area / (h * w) >= min_fill:
            best, best_area = sl, area
    return best


def find_label(img: Image.Image) -> Image.Image:
    """Locate the asset tag and crop to it: first a yellow tag, then a white
    sticker. Falls back to the full image if nothing tag-like is found."""
    rgb = img.convert("RGB")
    scale = min(1.0, 700 / max(rgb.size))
    small = rgb.resize((max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale)))) if scale < 1 else rgb
    a = np.asarray(small).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    yellow = (r > 170) & (g > 140) & (b < 110) & (r - b > 100) & (g - b > 70)
    white = (a.min(axis=2) > 170) & ((a.max(axis=2) - a.min(axis=2)) < 40)
    sl = _best_region(yellow, aspect=(1.6, 3.6)) or _best_region(white, min_frac=0.01)
    if sl is None:
        return img
    pad = 4
    y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, small.height)
    x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, small.width)
    return rgb.crop((int(x0 / scale), int(y0 / scale), int(x1 / scale), int(y1 / scale)))


def read_text(img: Image.Image, clean: bool = True) -> str:
    """clean=True: auto-crop to the label + image cleanup before OCR."""
    if clean:
        src = preprocess(find_label(img))
        text = pytesseract.image_to_string(src, config="--psm 6")
        if not re.search(r"S\s*[/I1L|]?\s*N\s*[:.\-]?\s*[0-9OQ]{4}", text.upper()):
            # tag not isolated (e.g. yellow tag on a yellow board): read the whole photo too
            text += "\n" + pytesseract.image_to_string(img.convert("RGB"), config="--psm 11")
        return text
    return pytesseract.image_to_string(img.convert("RGB"), config="--psm 11")


def parse_label(text: str, known_boards: list[str]) -> dict:
    """Pull board type, revision and serial number out of raw OCR text."""
    up = text.upper()
    serial = None
    m = re.search(r"S\s*[/I1L|]?\s*N\s*[:.\-]?\s*([0-9OQ]{4,6})", up)
    if m:
        serial = m.group(1).replace("O", "0").replace("Q", "0")
    rev = None
    m = re.search(r"REV\.?\s*([A-Z])\b", up)
    if m:
        rev = m.group(1)
    board = None
    for tok in re.findall(r"[A-Z0-9]{2,}\s*[-_]\s*[A-Z0-9]{2,}", up):
        tok = re.sub(r"\s*[-_]\s*", "-", tok)
        match = difflib.get_close_matches(tok, known_boards, n=1, cutoff=0.6)
        if not match:   # OCR often swaps letters for look-alike digits
            fixed = tok.translate(str.maketrans("0158", "OISB"))
            match = difflib.get_close_matches(fixed, known_boards, n=1, cutoff=0.6)
        if match:
            board = match[0]
            break
    return {"board_type": board, "revision": rev, "serial": serial}
