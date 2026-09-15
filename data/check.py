"""Verify retained inputs and report basic span and split checks."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Check file pins and every retained training span without rewriting data."""
    files = json.loads((ROOT / "local/reports/files.json").read_text())
    for name, expected in files.items():
        path = ROOT / name
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("Changed retained file: " + name)
    splits = {}
    types = set(json.loads((ROOT / "data/broad/types.json").read_text()))
    for folder in ("broad", "tune"):
        for path in sorted((ROOT / "data" / folder).glob("*.jsonl")):
            rows = 0
            texts = set()
            for line in path.read_text().splitlines():
                row = json.loads(line)
                text = row["text"]
                assert isinstance(text, str)
                for span in row["spans"]:
                    assert type(span["start"]) is int and type(span["end"]) is int
                    assert 0 <= span["start"] < span["end"] <= len(text)
                    assert span["type"] in types
                for start, end in row.get("ignore", []):
                    assert (
                        type(start) is int
                        and type(end) is int
                        and 0 <= start < end <= len(text)
                    )
                texts.add(hashlib.sha256(text.encode()).hexdigest())
                rows += 1
            splits[f"{folder}/{path.name}"] = {"rows": rows, "unique_texts": len(texts)}
    print(
        json.dumps(
            {
                "pinned_files": len(files),
                "splits": splits,
                "scope": "File integrity and span bounds only. This does not verify label truth or semantic independence.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
