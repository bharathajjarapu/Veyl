"""Train the pinned BERT teacher and distill only the selected Deep encoder."""

import hashlib
import json
import random
import time
from pathlib import Path
from .common import LABELS, TYPES, schema, prepare, training, teach, predict, pack


def main():
    """Run this fixed training stage into a new output directory."""
    import torch

    assert torch.cuda.is_available(), "CUDA GPU unavailable; no CPU training started"
    print({"gpu": torch.cuda.get_device_name(0)}, flush=True)
    import numpy
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        get_linear_schedule_with_warmup,
    )
    from torch.utils.data import DataLoader

    sample = torch.randn(1, 2, len(LABELS), requires_grad=True)
    teach(sample, torch.randn_like(sample), torch.tensor([[-100, 1]]), torch).backward()
    assert (
        torch.isfinite(sample.grad).all()
        and sample.grad[0, 0].eq(0).all()
        and (sample.grad[0, 1].abs().sum() > 0)
    ), "Distillation mask failed"
    random.seed(7)
    numpy.random.seed(7)
    torch.manual_seed(7)
    torch.set_num_threads(2)
    home = Path(__file__).resolve().parents[1]
    root = home / "runs" / "deep"
    root.mkdir(parents=True, exist_ok=False)
    source = home / "data" / "broad"
    config = json.loads((source / "job.json").read_text())
    schema(json.loads((source / "types.json").read_text()))
    model_specs = [["deep", "google/bert_uncased_L-6_H-128_A-2"]]
    teacher_base = config["teacher"]
    schedule, batches = (config["epochs"], config["batch"])
    raw = {
        split: [
            json.loads(line)
            for line in (source / (split + ".jsonl")).read_text().splitlines()
        ]
        for split in ("train", "valid", "test")
    }
    hashed_files = (
        "train.jsonl",
        "valid.jsonl",
        "test.jsonl",
        "data.json",
        "types.json",
        "job.json",
    )
    hashes = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in hashed_files
    }
    saved = json.loads((home / "local" / "reports" / "tagger.json").read_text())
    if hashes != saved["hashes"] or config != saved["config"]:
        raise ValueError("The selected Deep corpus or job changed")
    revisions = config.get("revisions", {})
    token_revision = revisions.get(model_specs[0][0], revisions.get(model_specs[0][1]))
    tokenizer = AutoTokenizer.from_pretrained(
        model_specs[0][1],
        use_fast=True,
        **{"revision": token_revision} if token_revision else {},
    )
    data, audit = ({}, {})
    for split, rows in raw.items():
        data[split], audit[split] = prepare(rows, tokenizer)
    report = {
        "data": json.loads((source / "data.json").read_text()),
        "audit": audit,
        "models": {},
        "scope": list(TYPES),
        "gpu": torch.cuda.get_device_name(0),
        "config": config,
        "hashes": hashes,
        "tokenizer": {
            "base": model_specs[0][1],
            "revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        },
        "seed": 7,
        "provisional": True,
    }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(
        {"audit": audit, "rows": {split: len(rows) for split, rows in data.items()}},
        flush=True,
    )
    collate = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
    entries = training(data["train"])
    mentor = None
    specs = [("mentor", teacher_base)] + [tuple(spec) for spec in model_specs]
    for name, base in specs:
        torch.manual_seed(7)
        rounds = schedule["teacher"] if name == "mentor" else schedule["student"]
        loader = DataLoader(
            entries,
            batch_size=batches["teacher"] if name == "mentor" else batches["student"],
            shuffle=True,
            collate_fn=collate,
        )
        path = root / name
        path.mkdir(exist_ok=True)
        revision = revisions.get(name, revisions.get(base))
        model = AutoModelForTokenClassification.from_pretrained(
            base,
            num_labels=len(LABELS),
            id2label=dict(enumerate(LABELS)),
            label2id={label: position for position, label in enumerate(LABELS)},
            attn_implementation="eager",
            **{"revision": revision} if revision else {},
        ).cuda()
        base_commit = getattr(model.config, "_commit_hash", None)
        report["models"][name] = {
            "base": base,
            "revision": base_commit,
            "provisional": True,
        }
        (root / "report.json").write_text(json.dumps(report, indent=2))
        assert (
            AutoTokenizer.from_pretrained(
                base, use_fast=True, revision=base_commit
            ).get_vocab()
            == tokenizer.get_vocab()
        ), "Teacher/student vocabulary differs"
        if name != "mentor":
            session = pack(model, tokenizer, path / "probe", torch)
            size = (path / "probe" / "model.onnx").stat().st_size
            print({"model": name, "probe": size, "fits": size <= 10000000}, flush=True)
            del session
            if size > 10000000:
                report["models"][name] = {
                    "base": base,
                    "revision": base_commit,
                    "bytes": size,
                    "fits": False,
                    "ready": False,
                    "provisional": True,
                }
                (root / "report.json").write_text(json.dumps(report, indent=2))
                continue
            model.cuda()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-05 if name == "mentor" else 0.0003,
            weight_decay=0.01,
        )
        scaler = torch.amp.GradScaler("cuda")
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(loader) // 2,
            num_training_steps=rounds * len(loader),
        )
        best, epochs = (-1.0, [])
        first = 0
        start = time.perf_counter()
        for epoch in range(first, rounds):
            model.train()
            loss = 0.0
            for step, batch in enumerate(loader):
                batch = {key: value.cuda() for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(**batch)
                    value = output.loss
                    if mentor is not None:
                        with torch.no_grad():
                            targets = mentor(
                                **{
                                    key: value
                                    for key, value in batch.items()
                                    if key != "labels"
                                }
                            ).logits
                        value = (
                            value
                            + teach(output.logits, targets, batch["labels"], torch)
                        ) / 2
                if not torch.isfinite(value):
                    raise ValueError("Non-finite training loss")
                scaler.scale(value).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale:
                    scheduler.step()
                loss += value.item()
                if step % 100 == 0:
                    print(
                        {
                            "model": name,
                            "epoch": epoch + 1,
                            "step": step,
                            "steps": len(loader),
                            "loss": loss / (step + 1),
                        },
                        flush=True,
                    )
            model.eval()
            with torch.inference_mode():
                result = predict(
                    data["valid"],
                    lambda batch: (
                        model(**{key: value.cuda() for key, value in batch.items()})
                        .logits.cpu()
                        .numpy()
                    ),
                    collate,
                    numpy,
                )
            epochs.append(
                {key: value for key, value in result.items() if key != "failures"}
            )
            print(
                {
                    "model": name,
                    "epoch": epoch + 1,
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "f1": result["f1"],
                    "macro": result["macro"],
                },
                flush=True,
            )
            selection = result["masking_f2"]
            if selection > best:
                best = selection
                model.save_pretrained(path / "best")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "rng": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(),
                },
                path / "checkpoint.pt",
            )
        del optimizer, scheduler, scaler
        if first < rounds:
            del batch, output, value
        if name == "mentor":
            del model
            mentor = (
                AutoModelForTokenClassification.from_pretrained(
                    path / "best", attn_implementation="eager"
                )
                .cuda()
                .eval()
                .requires_grad_(False)
            )
            report["models"][name] = {
                "base": base,
                "revision": base_commit,
                "epochs": epochs,
                "seconds": time.perf_counter() - start,
                "deployed": False,
            }
            (root / "report.json").write_text(json.dumps(report, indent=2))
            continue
        model = (
            AutoModelForTokenClassification.from_pretrained(
                path / "best", attn_implementation="eager"
            )
            .cuda()
            .eval()
        )
        with torch.inference_mode():
            native = predict(
                data["test"],
                lambda batch: (
                    model(**{key: value.cuda() for key, value in batch.items()})
                    .logits.cpu()
                    .numpy()
                ),
                collate,
                numpy,
            )
        session = pack(model, tokenizer, path, torch)
        run = lambda batch: session.run(
            None, {key: value.numpy() for key, value in batch.items()}
        )[0]
        quant = predict(data["test"], run, collate, numpy)
        challenge_rows = [row for row in data["test"] if row["source"] != "nvidia"]
        challenge = (
            predict(challenge_rows, run, collate, numpy) if challenge_rows else None
        )
        result = {
            "base": base,
            "revision": base_commit,
            "epochs": epochs,
            "float": native,
            "quant": quant,
            "challenge": challenge,
            "challenge_status": "evaluated"
            if challenge_rows
            else "not evaluated: no held-out challenge rows in this corpus",
            "distilled": mentor is not None,
            "bytes": (path / "model.onnx").stat().st_size,
            "tokenizer": (path / "tokenizer.json").stat().st_size,
            "seconds": time.perf_counter() - start,
            "parameters": sum((parameter.numel() for parameter in model.parameters())),
        }
        result["fits"] = result["bytes"] <= 10000000
        result["ready"] = False
        result["selection_metric"] = "masking_f2"
        report["models"][name] = result
        (root / "report.json").write_text(json.dumps(report, indent=2))
        print(
            {
                "model": name,
                "bytes": result["bytes"],
                "precision": quant["precision"],
                "recall": quant["recall"],
                "f1": quant["f1"],
                "ready": result["ready"],
            },
            flush=True,
        )
        del model, session
        torch.cuda.empty_cache()
    print("Deep training complete; outputs are in runs/deep.", flush=True)


if __name__ == "__main__":
    main()
