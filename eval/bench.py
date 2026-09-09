"""Benchmark the FP32 Deep model with gated AGE on PIIMB English."""

import argparse
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from eval.score import (
    EXPECTED_ROWS, SOURCE_ROWS, TASK_ROWS, TASKS, Score, check_manifest,
    rows as read_rows, sha, spans as checked_spans,
)
from src.model import ROOT, detect, load
from train.common import LABELS, prepare, spans, windows


def decode(row, tagged):
    """Keep detect's first-window tie break and most-central word prediction."""
    labels = [0] * len(row["words"])
    scores = [-1.0] * len(labels)
    for (lower, upper, start, end, entry), tags in tagged:
        if len(tags) != len(entry["input_ids"]):
            raise ValueError("Window prediction length mismatch")
        for word in range(lower, upper):
            token = row["heads"][word]
            central = min(
                math.inf if start == 0 else token - start,
                math.inf if end == len(row["ids"]) else end - 1 - token,
            )
            if central > scores[word]:
                labels[word], scores[word] = tags[token - start + 1], central
    if any(score < 0 for score in scores):
        raise ValueError("A word has no window prediction")
    return spans(row["words"], labels)


def predict(texts, model, tokenizer, device, batch_size, torch):
    """Batch similar-length windows, then decode in their original order."""
    prepared, skipped = prepare(
        [{"text": text, "spans": []} for text in texts], tokenizer
    )
    if skipped or len(prepared) != len(texts):
        raise ValueError("Inference preparation skipped rows: " + str(skipped))
    entries = [(doc, entry) for doc, row in enumerate(prepared) for entry in windows(row)]
    tags = [None] * len(entries)
    ordered = sorted(range(len(entries)), key=lambda index: len(entries[index][1][-1]["input_ids"]))
    with torch.inference_mode():
        for offset in range(0, len(ordered), batch_size):
            indices = ordered[offset:offset + batch_size]
            data = tokenizer.pad(
                [{name: value for name, value in entries[index][1][-1].items()
                  if name != "labels"} for index in indices],
                padding=True, return_tensors="pt",
            )
            logits = model(**{name: value.to(device) for name, value in data.items()})
            if (tuple(logits.shape) != (*data["input_ids"].shape, len(LABELS))
                    or logits.dtype != torch.float32 or not torch.isfinite(logits).all()):
                raise ValueError("Invalid FP32 model output")
            predicted = logits.argmax(-1).cpu().tolist()
            for index, values in zip(indices, predicted, strict=True):
                tags[index] = values[:len(entries[index][1][-1]["input_ids"])]
    tagged = [[] for unused in prepared]
    for (doc, entry), values in zip(entries, tags, strict=True):
        tagged[doc].append((entry, values))
    return [decode(row, values) for row, values in zip(prepared, tagged, strict=True)], len(entries)


def self_check():
    """Exercise overlap ownership, ties, empty text, BIO merging and F2."""
    row = {"ids": list(range(700)), "heads": list(range(700)),
           "words": [[index * 2, index * 2 + 1] for index in range(700)],
           "tags": [0] * 700, "cls": 101, "sep": 102}
    entries = windows(row)
    assert len(entries) == 2
    first = [1] * len(entries[0][-1]["input_ids"])
    last = [3] * len(entries[1][-1]["input_ids"])
    found = decode(row, [(entries[0], first), (entries[1], last)])
    assert len(found) == 700 and found[477]["type"] == "PERSON" and found[478]["type"] == "LOCATION"
    assert decode(row, [(entries[0], first), (entries[0], [3] * len(first)),
                        (entries[1], last)]) == found
    assert decode({"words": []}, []) == []
    assert spans([[0, 4], [5, 10]], [1, 2]) == [{"start": 0, "end": 10, "type": "PERSON"}]
    score = Score()
    score.add("abc", [{"start": 0, "end": 3}], [{"start": 0, "end": 2}])
    assert abs(score.result()["f2"] - 10 / 11) < 1e-12


def main():
    """Retain predictions and a separately identified full FP32 benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/bench/fp32")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.check:
        print("FP32 benchmark decoder self-check passed")
        return
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    import torch

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable; use --device cpu")
    source = ROOT / "data/eval/piimb.jsonl"
    source_hash, unused = check_manifest("piimb", source)
    pins = json.loads((ROOT / "local/reports/files.json").read_text())
    names = ("weights/deep/model.safetensors", "weights/deep/config.json",
             "weights/head.pt", "weights/tokenizer.json", "weights/labels.json",
             "weights/tokens.json", "data/eval/piimb.jsonl", "local/reports/piimb.json")
    hashes = {name: sha(ROOT / name) for name in names}
    if any(hashes[name] != pins[name] for name in names):
        raise ValueError("Benchmark input differs from its retained hash")
    if json.loads((ROOT / "weights/labels.json").read_text()) != LABELS:
        raise ValueError("Label schema differs from the decoder")
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite benchmark output: " + str(args.out))
    model, tokenizer = load()
    model = model.float().to(device).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == 5_170_107
    fixtures = json.loads((ROOT / "weights/tokens.json").read_text())
    for fixture in fixtures:
        prepared, skipped = prepare([{"text": fixture["text"], "spans": []}], tokenizer)
        assert not skipped and prepared[0]["ids"] == fixture["ids"] and prepared[0]["words"] == fixture["words"]
    probes = [fixture["text"] for fixture in fixtures] + ["", "Contact John Smith.",
              "a " * 600 + "Contact John Smith."]
    batched, unused = predict(probes, model, tokenizer, device, args.batch_size, torch)
    single = lambda **inputs: model(**{name: value.to(device) for name, value in inputs.items()})
    assert batched == [detect(text, single, tokenizer) for text in probes], "Batch/detect fixture predictions differ"

    args.out.mkdir(parents=True)
    prediction = args.out / "piimb.jsonl"
    selected, tasks, counts = Score(), {task: Score() for task in TASKS}, Counter()
    chunk, source_rows, window_count = [], 0, 0
    started = time.perf_counter()

    def flush(stream):
        """Score and persist each bounded chunk in selected-source order."""
        nonlocal window_count
        found, count = predict([row["text"] for number, row in chunk], model, tokenizer,
                               device, args.batch_size, torch)
        window_count += count
        for (number, row), predicted in zip(chunk, found, strict=True):
            task = row["task_name"]
            gold = checked_spans(row, "entities")
            selected.add(row["text"], predicted, gold)
            tasks[task].add(row["text"], predicted, gold)
            counts[task] += 1
            stream.write(json.dumps({"row": selected.rows, "source_row": number,
                                    "id": str(row["uid"]), "task": task,
                                    "spans": predicted}, ensure_ascii=False) + "\n")
        chunk.clear()
        print(f"{selected.rows}/{EXPECTED_ROWS['piimb']} rows; {time.perf_counter() - started:.1f}s", flush=True)

    with prediction.open("x", encoding="utf-8") as stream:
        for source_rows, row in enumerate(read_rows(source), 1):
            if row.get("language") == "en" and row.get("task_name") in TASKS:
                if not isinstance(row.get("text"), str) or not isinstance(row.get("uid"), str):
                    raise ValueError(f"Invalid source row {source_rows}")
                chunk.append((source_rows, row))
                if len(chunk) == 1024:
                    flush(stream)
        if chunk:
            flush(stream)
    if source_rows != SOURCE_ROWS or selected.rows != EXPECTED_ROWS["piimb"] or dict(counts) != TASK_ROWS:
        raise ValueError("PIIMB source or English task counts changed")
    elapsed = time.perf_counter() - started
    meta = {
        "model": "veyl-fp32", "dataset": "piimb", "selection": "piimb",
        "source_sha256": source_hash, "output_sha256": sha(prediction),
        "input_sha256": hashes, "parameters": parameters, "rows": selected.rows,
        "source_rows": source_rows, "task_rows": dict(counts), "windows": window_count,
        "window": 510, "overlap": 64, "minimum_confidence": 0,
        "decoder": "First subword; most-central window; first-window ties; BIO spans; code point offsets",
        "decoder_note": "Python Hugging Face tokenizer; retained fixtures checked. Full custom WASM tokenizer equivalence is not established.",
        "batch_detect_fixture_rows": len(probes),
        "runtime_sha256": {name: sha(ROOT / name) for name in
                           ("eval/bench.py", "eval/score.py", "src/model.py", "train/common.py")},
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "torch": torch.__version__, "transformers": version("transformers"),
                    "tokenizers": version("tokenizers"), "cuda": torch.version.cuda,
                    "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                    "dtype": "float32", "tf32": False, "autocast": False,
                    "batch_size": args.batch_size, "torch_threads": torch.get_num_threads()},
        "seconds": elapsed,
        "timing": "Batched wall time includes tokenization, inference, decoding, scoring and prediction writes; excludes load and startup checks. Not per-row latency.",
    }
    per_task = {task: tasks[task].result() for task in TASKS}
    result = {
        "model": "veyl-fp32", "dataset": "piimb", "rows": selected.rows,
        "source_sha256": source_hash, "prediction_sha256": meta["output_sha256"],
        "scorer_sha256": sha(ROOT / "eval/score.py"), "model_only": True,
        "tasks": per_task, "micro": selected.result(), "task_order": list(TASKS),
        "unweighted_task_f2_mean": statistics.mean(per_task[task]["f2"] for task in TASKS),
        "unweighted_task_precision_mean": statistics.mean(per_task[task]["precision"] for task in TASKS),
        "unweighted_task_recall_mean": statistics.mean(per_task[task]["recall"] for task in TASKS),
        "prediction_meta": meta,
        "metric_note": "Local PIIMB 0.3.0 label-agnostic character-mask calculation. Not an official leaderboard result. GPU FP32 and INT8 WASM use different numerical/runtime paths.",
    }
    prediction.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (args.out / "score.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
