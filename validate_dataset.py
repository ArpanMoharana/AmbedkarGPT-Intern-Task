# validate_dataset.py
import json
from pathlib import Path
import sys

TEST_FILE = "test_dataset.json"
CORPUS_DIR = Path("corpus")

def main():
    if not Path(TEST_FILE).exists():
        print(f"Error: {TEST_FILE} not found.")
        sys.exit(1)
    if not CORPUS_DIR.exists():
        print(f"Error: corpus directory '{CORPUS_DIR}' not found.")
        sys.exit(1)

    data = json.load(open(TEST_FILE, "r", encoding="utf-8"))
    questions = data.get("test_questions", [])
    missing = {}
    for q in questions:
        srcs = q.get("source_documents", [])
        for s in srcs:
            p = CORPUS_DIR / s
            if not p.exists():
                missing.setdefault(s, []).append(q.get("id"))

    if not missing:
        print("All referenced source_documents exist in corpus/.")
    else:
        print("Missing files referenced in test_dataset.json:")
        for fn, qids in missing.items():
            print(f"  - {fn} referenced by question ids: {qids}")
        sys.exit(2)

if __name__ == "__main__":
    main()
