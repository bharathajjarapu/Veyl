"""Load the retained FP32 model for Python inference on Kaggle."""

import hashlib
import json
import math
import sys
from pathlib import Path

from train.common import LABELS, prepare, spans, windows

ROOT = Path(__file__).resolve().parents[1]
BEGIN = LABELS.index("B-AGE")
INSIDE = LABELS.index("I-AGE")


def load():
    """Load local Deep and AGE weights with the accepted O-only gate."""
    import torch
    from transformers import AutoModelForTokenClassification, BertTokenizerFast

    pins = json.loads((ROOT / "local/reports/files.json").read_text())
    for name in (
        "deep/model.safetensors",
        "deep/config.json",
        "head.pt",
        "tokenizer.json",
        "labels.json",
    ):
        path = ROOT / "weights" / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != pins["weights/" + name]:
            raise ValueError("Changed model input: " + name)
    base = AutoModelForTokenClassification.from_pretrained(
        ROOT / "weights/deep", local_files_only=True, attn_implementation="eager"
    )
    if base.config.num_labels != len(LABELS) or base.config.id2label != dict(
        enumerate(LABELS)
    ):
        raise ValueError("Deep label order differs from the selected schema")
    with torch.no_grad():
        base.classifier.bias[0] -= 1.0

    class Model(torch.nn.Module):
        def __init__(self):
            """Attach the retained encoder and selected age head."""
            super().__init__()
            self.base = base
            self.head = torch.nn.Linear(base.config.hidden_size, 2)
            self.head.load_state_dict(
                torch.load(
                    ROOT / "weights/head.pt", map_location="cpu", weights_only=True
                )
            )

        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            """Add the age residual only where Deep predicts O."""
            hidden = self.base.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=False,
            )[0]
            logits = self.base.classifier(self.base.dropout(hidden))
            delta = torch.nn.functional.softplus(self.head(hidden))
            delta = torch.where(logits.argmax(-1, keepdim=True) == 0, delta, 0.0)
            return torch.cat(
                (
                    logits[..., :BEGIN],
                    logits[..., BEGIN : INSIDE + 1] + delta,
                    logits[..., INSIDE + 1 :],
                ),
                dim=-1,
            )

    tokenizer = BertTokenizerFast(tokenizer_file=str(ROOT / "weights/tokenizer.json"))
    return Model().requires_grad_(False).eval(), tokenizer


def detect(text, model, tokenizer):
    """Return code point spans using the shared word-aligned windows."""
    import torch

    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    rows, skipped = prepare([{"text": text, "spans": []}], tokenizer)
    if skipped or len(rows) != 1:
        raise ValueError("Text could not be prepared: " + str(skipped))
    row = rows[0]
    labels = [0] * len(row["words"])
    scores = [-1.0] * len(labels)
    with torch.inference_mode():
        for lower, upper, start, end, entry in windows(row):
            inputs = {
                name: torch.tensor([value], dtype=torch.long)
                for name, value in entry.items()
                if name != "labels"
            }
            logits = model(**inputs)
            if (
                logits.shape != (1, len(entry["input_ids"]), len(LABELS))
                or not torch.isfinite(logits).all()
            ):
                raise ValueError("Invalid model output")
            tags = logits[0].argmax(-1).tolist()
            for word in range(lower, upper):
                token = row["heads"][word]
                central = min(
                    math.inf if start == 0 else token - start,
                    math.inf if end == len(row["ids"]) else end - 1 - token,
                )
                if central > scores[word]:
                    labels[word], scores[word] = tags[token - start + 1], central
    return spans(row["words"], labels)


if __name__ == "__main__":
    if not sys.argv[1:]:
        raise SystemExit("Usage: python -m src.model 'text'")
    model, tokenizer = load()
    print(
        json.dumps(
            {"spans": detect(" ".join(sys.argv[1:]), model, tokenizer)},
            ensure_ascii=False,
            indent=2,
        )
    )
