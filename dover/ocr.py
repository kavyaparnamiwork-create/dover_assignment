"""OCR backends. Anything returning list[Word] with 0..1 confidences can be plugged in."""
import cv2
import pytesseract
from pytesseract import Output

from .nameplate_types import Word


class TesseractOCR:
    def __init__(self, page_psm=11, lang="eng"):
        self.page_psm, self.lang = page_psm, lang

    def _run(self, img, psm):
        d = pytesseract.image_to_data(
            img, lang=self.lang, config=f"--oem 1 --psm {psm}", output_type=Output.DICT
        )
        words = []
        for t, c, x, y, w, h in zip(d["text"], d["conf"], d["left"], d["top"], d["width"], d["height"]):
            t = t.strip()
            if t and float(c) >= 0:
                words.append(Word(t, float(c) / 100.0, x, y, w, h))
        return words

    def read_page(self, img):
        return self._run(img, self.page_psm)

    def read_crop(self, crop):
        """Single-line read of a cropped value box."""
        return self._run(crop, 7)


def order_words(words):
    """Reading order: cluster by row, then left to right."""
    if not words:
        return []
    tol = sorted(w.h for w in words)[len(words) // 2] * 0.6
    rows = []
    for w in sorted(words, key=lambda w: w.cy):
        if rows and abs(w.cy - rows[-1]["cy"]) <= tol:
            rows[-1]["w"].append(w)
            rows[-1]["cy"] = sum(x.cy for x in rows[-1]["w"]) / len(rows[-1]["w"])
        else:
            rows.append({"cy": w.cy, "w": [w]})
    return [w for r in rows for w in sorted(r["w"], key=lambda w: w.x)]


def group_lines(words):
    """Same clustering, but keeps the rows."""
    ordered = order_words(words)
    if not ordered:
        return []
    tol = sorted(w.h for w in words)[len(words) // 2] * 0.6
    lines, cur = [], [ordered[0]]
    for w in ordered[1:]:
        if abs(w.cy - sum(x.cy for x in cur) / len(cur)) <= tol:
            cur.append(w)
        else:
            lines.append(cur)
            cur = [w]
    lines.append(cur)
    return lines


def prep_crop(gray_full, box, scale, pad=8, min_h=72, binarize=False):
    x, y, w, h = [v / scale for v in box]
    x0, y0 = max(int(x - pad), 0), max(int(y - pad), 0)
    x1, y1 = min(int(x + w + pad), gray_full.shape[1]), min(int(y + h + pad), gray_full.shape[0])
    crop = gray_full[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    f = max(min_h / crop.shape[0], 1.0)
    crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    if binarize:
        _, crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.copyMakeBorder(crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
