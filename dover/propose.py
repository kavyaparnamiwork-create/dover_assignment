"""Step 2: propose open-ended key/value pairs. Proposals are untrusted; verify.py decides."""
import base64
import json
import re
import urllib.request

import cv2

from .ocr import group_lines
from .nameplate_types import Proposal

_HAS_DIGIT = re.compile(r"\d")


class LayoutProposer:
    """No-model fallback: on each OCR row, the label is the leading words and the value starts
    at the first word containing a digit (or follows a ':'). Weaker on wrapped labels."""

    def propose(self, work_gray, words):
        out = []
        for line in group_lines(words):
            toks = [w.text for w in line]
            joined = " ".join(toks)
            if ":" in joined:
                k, _, v = joined.partition(":")
            else:
                i = next((i for i, t in enumerate(toks) if _HAS_DIGIT.search(t)), None)
                if i is None or i == 0:
                    continue
                k, v = " ".join(toks[:i]), " ".join(toks[i:])
            k, v = k.strip(" :-"), v.strip()
            if k and v:
                out.append(Proposal(k, v))
        return out


PROMPT = (
    "This is a photo of a metal equipment nameplate. List every printed label together with the "
    "value stamped or printed next to it. Return only JSON: "
    '{"fields":[{"key":"<label as printed>","value":"<value as printed>"}]}. '
    "Include as many pairs as are on the plate. Copy text exactly as printed. "
    "Skip anything covered, cut off, rusted or unclear. Never guess."
)


class OllamaProposer:
    """Local vision-language model served by Ollama on localhost (no internet needed once the
    model is pulled). Example: `ollama pull qwen2.5vl:7b`."""

    def __init__(self, model="qwen2.5vl:7b", host="http://127.0.0.1:11434", timeout=600):
        self.model, self.host, self.timeout = model, host, timeout

    def propose(self, work_gray, words):
        ok, png = cv2.imencode(".png", work_gray)
        body = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [{"role": "user", "content": PROMPT,
                          "images": [base64.b64encode(png.tobytes()).decode()]}],
        }
        req = urllib.request.Request(self.host + "/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            content = json.loads(r.read())["message"]["content"]
        data = json.loads(content)
        items = data.get("fields", []) if isinstance(data, dict) else data
        return [Proposal(str(i["key"]).strip(), str(i["value"]).strip())
                for i in items if isinstance(i, dict) and i.get("key") and i.get("value")]
