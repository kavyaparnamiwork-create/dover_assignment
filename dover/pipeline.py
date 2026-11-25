from .ocr import TesseractOCR
from .prepare import load_image, prepare
from .propose import LayoutProposer
from .verify import verify


def extract(path, proposer=None, ocr=None, min_conf=0.80, debug=False):
    ocr = ocr or TesseractOCR()
    proposer = proposer or LayoutProposer()
    gray_full, work, scale = prepare(load_image(path))
    words = ocr.read_page(work)
    proposals = proposer.propose(work, words)
    verdicts = verify(proposals, words, gray_full, scale, ocr, min_conf)
    result = {
        "source": str(path).split("/")[-1],
        "fields": [{"key": v.key, "value": v.value, "confidence": v.confidence}
                   for v in verdicts if v.kept],
    }
    if debug:
        result["rejected"] = [{"key": v.key, "value": v.value, "reason": v.reason,
                               "confidence": round(v.confidence, 3)} for v in verdicts if not v.kept]
    return result
