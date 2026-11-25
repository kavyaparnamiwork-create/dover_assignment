import argparse
import json
from pathlib import Path

from .pipeline import extract
from .propose import LayoutProposer, OllamaProposer

EXTS = {".png", ".jpg", ".jpeg", ".heic"}


def main():
    ap = argparse.ArgumentParser(description="Nameplate -> verified key/value JSON (offline)")
    ap.add_argument("input", help="image file or folder")
    ap.add_argument("-o", "--output", help="write JSON here (default: stdout)")
    ap.add_argument("--proposer", choices=["layout", "ollama"], default="layout")
    ap.add_argument("--model", default="qwen2.5vl:7b", help="Ollama vision model name")
    ap.add_argument("--min-conf", type=float, default=0.80)
    ap.add_argument("--debug", action="store_true", help="include rejected pairs and reasons")
    a = ap.parse_args()

    proposer = LayoutProposer() if a.proposer == "layout" else OllamaProposer(model=a.model)
    p = Path(a.input)
    files = sorted(f for f in p.iterdir() if f.suffix.lower() in EXTS) if p.is_dir() else [p]
    results = [extract(f, proposer, min_conf=a.min_conf, debug=a.debug) for f in files]
    out = json.dumps(results[0] if len(results) == 1 else results, indent=2, ensure_ascii=False)
    if a.output:
        Path(a.output).write_text(out)
    else:
        print(out)


if __name__ == "__main__":
    main()
