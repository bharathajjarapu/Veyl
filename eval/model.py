"""Check the retained FP32 model on a machine with PyTorch, such as Kaggle."""

import json
from src.model import BEGIN, INSIDE, ROOT, detect, load
from train import age, common


def main():
    """Check real weights, gradients, token fixtures and the inference gate."""
    import torch

    torch.set_num_threads(2)
    assert age.losscheck(torch) and age.headcheck(torch)
    logits = torch.randn(1, 2, len(common.LABELS), requires_grad=True)
    common.teach(
        logits, torch.randn_like(logits), torch.tensor([[-100, 1]]), torch
    ).backward()
    assert logits.grad[0, 0].eq(0).all() and torch.isfinite(logits.grad).all()
    model, tokenizer = load()
    assert sum(parameter.numel() for parameter in model.parameters()) == 5170107
    fixtures = json.loads((ROOT / "weights/tokens.json").read_text())
    with torch.inference_mode():
        for row in fixtures:
            prepared, skipped = common.prepare(
                [{"text": row["text"], "spans": []}], tokenizer
            )
            assert (
                not skipped
                and prepared[0]["ids"] == row["ids"]
                and prepared[0]["words"] == row["words"]
            )
            inputs = tokenizer(row["text"], return_tensors="pt")
            old = model.base(**inputs).logits
            new = model(**inputs)
            original = age.head(torch, model.base).eval()
            original.head.load_state_dict(model.head.state_dict())
            candidate = original(**inputs).logits
            expected = torch.where(old.argmax(-1, keepdim=True) == 0, candidate, old)
            assert torch.equal(new, expected)
            assert torch.equal(old[..., :BEGIN], new[..., :BEGIN])
            assert torch.equal(old[..., INSIDE + 1 :], new[..., INSIDE + 1 :])
            assert torch.isfinite(new).all()
    text = "Contact John Smith at john@example.com."
    found = detect(text, model, tokenizer)
    assert any(
        text[span["start"] : span["end"]] == "John Smith" and span["type"] == "PERSON"
        for span in found
    )
    for text in ("", "😀 中文 John Smith", "a " * 600 + text):
        found = detect(text, model, tokenizer)
        assert all(0 <= span["start"] < span["end"] <= len(text) for span in found)
    print(
        json.dumps(
            {
                "passed": True,
                "parameters": 5170107,
                "token_fixtures": len(fixtures),
                "runtime": "PyTorch FP32 CPU",
                "torch": torch.__version__,
                "scope": "Model, gating and gradient smoke checks. No training or benchmark score was measured.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
