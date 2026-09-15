"""Run CPU checks for selected training boundaries without starting training."""

import ast
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy

from . import age, common, deep


def main():
    """Check window merging, coverage masks, AGE gates and import boundaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--torch",
        action="store_true",
        help="also check CPU PyTorch loss and head gradients",
    )
    options = parser.parse_args()
    assert age.losscheck() and age.exportcheck(numpy) and age.datecheck()
    if options.torch:
        import torch

        assert age.losscheck(torch) and age.headcheck(torch)
        logits = torch.randn(1, 2, len(common.LABELS), requires_grad=True)
        common.teach(
            logits, torch.randn_like(logits), torch.tensor([[-100, 1]]), torch
        ).backward()
        assert torch.isfinite(logits.grad).all()
        assert logits.grad[0, 0].eq(0).all() and logits.grad[0, 1].abs().sum() > 0
    home = Path(__file__).resolve().parents[1]
    source = home / "data/tune"
    manifest = json.loads((source / "data.json").read_text())
    for name in age.PROTOCOL["input_files"]:
        path = age.corpus(source, name)
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == manifest["input_hashes"][name]
        ), name
    assert age.corpus(source, "nv.train.jsonl") == home / "data/broad/train.jsonl"
    report = json.loads((home / "local" / "reports" / "age.json").read_text())
    baseline = report["baseline"]
    macro = report["baseline_macro_source_char_f2"]
    for entry in report["history"]:
        assert age.eligible(entry["metrics"], baseline, macro) == entry["eligible"]
    chosen = max(
        (entry for entry in report["history"] if entry["eligible"]),
        key=lambda entry: entry["macro_source_char_f2"],
    )
    assert chosen["step"] == 900 and report["selected_head_bias_delta"] == 0
    tagger = json.loads((home / "local" / "reports" / "tagger.json").read_text())
    for name, digest in tagger["hashes"].items():
        assert (
            hashlib.sha256((home / "data" / "broad" / name).read_bytes()).hexdigest()
            == digest
        ), name
    assert len(common.TYPES) == 28 and len(common.LABELS) == 57
    common.schema(common.TYPES)
    try:
        common.schema(common.TYPES[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("Changed taxonomy was accepted")
    row = {
        "text": "Jane 2034",
        "spans": [
            {"start": 0, "end": 4, "type": "PERSON"},
            {"start": 5, "end": 9, "type": "DATE_TIME"},
        ],
        "coverage": list(common.TYPES),
        "ignore": [],
        "words": [[0, 4], [5, 9]],
        "tags": [1, 15],
        "heads": [0, 1],
        "ids": [1010, 1011],
        "cls": 101,
        "sep": 102,
        "source": "nv",
    }
    window = common.windows(row)
    assert len(window) == 1 and window[0][-1]["labels"] == [-100, 1, 15, -100]
    assert common.spans(row["words"], row["tags"]) == row["spans"]
    wrong = age.evaluate([row], [[{"start": 0, "end": 4, "type": "AGE"}]], [[17, 0]])
    assert (
        wrong["strict_age"]["fp"] == 1
        and wrong["strict_age"]["fully_negative_haiku_age_fp_spans"] == 0
    )
    annotated = age.annotate([copy.deepcopy(row)], "luna.train")[0]
    assert annotated["ignore"] == [[4, 5]]
    entries = age.entries([row])
    assert entries[0]["_negative"] == sorted(common.TYPES)
    long = dict(row, ids=list(range(900)), heads=list(range(900)), tags=[0] * 900)
    windows = common.windows(long)
    assert all(len(window[-1]["input_ids"]) <= 512 for window in windows)
    assert windows[0][1] > windows[1][0] and windows[-1][3] == 900
    try:
        common.windows(dict(row, ids=list(range(511)), heads=[0], tags=[0]))
    except ValueError:
        pass
    else:
        raise AssertionError("Oversize word was accepted")
    for module in (common, age, deep):
        tree = ast.parse(Path(module.__file__).read_text())
        functions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                called = {
                    child.func.id
                    for child in ast.walk(node)
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                }
                stored = {
                    child.id
                    for child in ast.walk(node)
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
                }
                assert not functions & called & stored, node.name
    print("Training CPU checks passed; no training was started.")


if __name__ == "__main__":
    main()
