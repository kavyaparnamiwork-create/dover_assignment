"""Step 4: the confidence gate. A pair survives only if
  1. its key is found in the page OCR,
  2. its value is found in the page OCR shortly after the key,
  3. a second, independent read of the value's cropped box gives the same text,
  4. every confidence involved is >= min_conf.
Nothing is corrected or filled in. Anything that fails is dropped (with a reason for --debug)."""
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .ocr import order_words, prep_crop

MAX_GAP_WORDS = 10


def norm(s):
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def _find(words, target, min_ratio, exact, start=0, stop=None):
    """Best run of consecutive words whose normalised text matches target."""
    stop = len(words) if stop is None else stop
    best = None
    for i in range(start, stop):
        acc = ""
        for j in range(i, stop):
            acc += norm(words[j].text)
            if len(acc) > len(target) + 3:
                break
            if not acc:
                continue
            r = 1.0 if acc == target else (0.0 if exact else SequenceMatcher(None, acc, target).ratio())
            if r >= min_ratio and (best is None or r > best[0]):
                best = (r, i, j + 1)
    return best


@dataclass
class Verdict:
    key: str
    value: str
    kept: bool
    confidence: float = 0.0
    reason: str = ""


def verify(proposals, words, gray_full, scale, ocr, min_conf=0.80):
    words = order_words(words)
    out, seen = [], set()
    for p in proposals:
        tk, tv = norm(p.key), norm(p.value)
        if not tk or not tv:
            out.append(Verdict(p.key, p.value, False, reason="empty"))
            continue
        if (tk, tv) in seen:
            continue
        seen.add((tk, tv))
        exact_value = any(c.isdigit() for c in tv)  # numbers must match character for character

        km = _find(words, tk, 0.85, exact=False)
        if not km:
            out.append(Verdict(p.key, p.value, False, reason="key not found in OCR"))
            continue
        _, ks, ke = km
        vm = _find(words, tv, 0.90, exact=exact_value, start=ke, stop=min(ke + MAX_GAP_WORDS, len(words)))
        if not vm:
            out.append(Verdict(p.key, p.value, False, reason="value not found in OCR after key"))
            continue
        _, vs, ve = vm
        kconf = min(w.conf for w in words[ks:ke])
        vwords = words[vs:ve]
        vconf = min(w.conf for w in vwords)
        if kconf < min_conf:
            out.append(Verdict(p.key, p.value, False, min(kconf, vconf), "key confidence low"))
            continue

        # independent second read of the value's box; one retry with a different preprocessing
        x0 = min(w.x for w in vwords); y0 = min(w.y for w in vwords)
        x1 = max(w.x + w.w for w in vwords); y1 = max(w.y + w.h for w in vwords)
        box = (x0, y0, x1 - x0, y1 - y0)
        crop_conf, agreed = 0.0, False
        for binarize in (False, True):
            crop = prep_crop(gray_full, box, scale, binarize=binarize)
            if crop is None:
                break
            cw = ocr.read_crop(crop)
            if not cw:
                continue
            read = "".join(norm(w.text) for w in cw)
            crop_conf = min(w.conf for w in cw)
            agreed = (read == tv) if exact_value else SequenceMatcher(None, read, tv).ratio() >= 0.90
            if agreed and crop_conf >= min_conf:
                break
        if not agreed:
            out.append(Verdict(p.key, p.value, False, min(kconf, vconf), "crop re-read disagrees"))
            continue
        # first read below threshold is rescued only by the retry crop; otherwise both reads count
        value_conf = crop_conf if vconf < min_conf else min(vconf, crop_conf)
        conf = min(kconf, value_conf)
        if conf < min_conf:
            out.append(Verdict(p.key, p.value, False, conf, "confidence below threshold"))
            continue
        out.append(Verdict(p.key, p.value, True, round(conf, 3)))
    return out
