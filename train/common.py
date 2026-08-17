"""Shared token preparation and INT8 export used by both training stages."""

import json
import math
import time
from collections import Counter

TYPES = (
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
    "PASSWORD",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "URL",
    "DATE_TIME",
    "AGE",
    "US_SSN",
    "NATIONAL_ID",
    "TAX_ID",
    "BANK_ACCOUNT",
    "CREDIT_CARD",
    "CARD_SECURITY_CODE",
    "USERNAME",
    "MEDICAL_RECORD_ID",
    "CUSTOMER_ID",
    "EMPLOYEE_ID",
    "OTHER_ID",
    "VEHICLE_ID",
    "IP_ADDRESS",
    "DEVICE_ID",
    "API_KEY",
    "HTTP_COOKIE",
    "COORDINATE",
    "BIOMETRIC_IDENTIFIER",
    "SENSITIVE_ATTRIBUTE",
)
LABELS = ["O"] + [prefix + "-" + kind for kind in TYPES for prefix in ("B", "I")]


def schema(types):
    """Require the fixed 28-type schema without changing imported label references."""
    if tuple(types) != TYPES:
        raise ValueError("Expected the selected 28-type schema")


def ignored(row, start, end):
    """Check whether an offset range overlaps ignored text."""
    return any((left < end and start < right for left, right in row.get("ignore", [])))


def windows(row):
    """Build overlapping first-subword BIO windows of at most 512 tokens."""
    ids, heads = (row["ids"], row["heads"])
    out, lower = ([], 0)
    while lower < len(heads):
        start, upper = (heads[lower], lower + 1)
        while (
            upper < len(heads)
            and (heads[upper + 1] if upper + 1 < len(heads) else len(ids)) - start
            <= 510
        ):
            upper += 1
        end = heads[upper] if upper < len(heads) else len(ids)
        if end - start > 510:
            raise ValueError("Word exceeds window")
        tags = [-100] * (end - start + 2)
        for word in range(lower, upper):
            tags[heads[word] - start + 1] = row["tags"][word]
        data = {
            "input_ids": [row["cls"]] + ids[start:end] + [row["sep"]],
            "attention_mask": [1] * (end - start + 2),
            "token_type_ids": [0] * (end - start + 2),
            "labels": tags,
        }
        out.append((lower, upper, start, end, data))
        if end == len(ids):
            break
        step = lower
        while step + 1 < upper and heads[step + 1] <= end - 64:
            step += 1
        lower = max(lower + 1, step)
    return out


def training(rows):
    """Collect supervised windows for teacher and Deep training."""
    return [
        entry[-1]
        for row in rows
        for entry in windows(row)
        if any((tag != -100 for tag in entry[-1]["labels"]))
    ]


def prepare(rows, tokenizer):
    """Tokenize rows and reject spans that do not match word boundaries."""
    out, skipped = ([], Counter())
    for row in rows:
        text, spans = (row["text"], row["spans"])
        for start, end in row.get("ignore", []):
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or (not 0 <= start < end <= len(text))
            ):
                raise ValueError("Invalid ignore range")
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        words, heads, parts = ([], [], encoded.word_ids())
        for position, word in enumerate(parts):
            start, end = encoded["offset_mapping"][position]
            if word is None:
                continue
            if not heads or word != parts[heads[-1]]:
                heads.append(position)
                words.append([start, end])
            else:
                words[-1][1] = end
        tags, valid = ([0] * len(words), True)
        for position, (start, end) in enumerate(words):
            if ignored(row, start, end):
                tags[position] = -100
        for span in spans:
            start, end, kind = (span["start"], span["end"], span["type"])
            hits = [
                position
                for position, word in enumerate(words)
                if word[0] < end and start < word[1]
            ]
            if (
                kind not in TYPES
                or not hits
                or words[hits[0]][0] != start
                or (words[hits[-1]][1] != end)
            ):
                valid = False
                break
            active = [position for position in hits if tags[position] != -100]
            for position in active:
                if tags[position] not in (0, -100):
                    valid = False
                    break
                if tags[position] != -100:
                    tags[position] = LABELS.index(
                        ("B-" if position == active[0] else "I-") + kind
                    )
        if not valid:
            skipped["boundary"] += 1
            continue
        if tags and all((tag == -100 for tag in tags)):
            skipped["ignored"] += 1
            continue
        item = dict(
            row,
            ids=encoded["input_ids"],
            heads=heads,
            words=words,
            tags=tags,
            cls=tokenizer.cls_token_id,
            sep=tokenizer.sep_token_id,
        )
        try:
            windows(item)
        except ValueError:
            skipped["oversize"] += 1
            continue
        out.append(item)
    return (out, dict(skipped))


def spans(words, tags):
    """Decode word BIO labels into offset spans."""
    if len(words) != len(tags):
        raise ValueError("Token prediction length mismatch")
    out, current = ([], None)
    for word, tag in zip(words, tags):
        label = LABELS[int(tag)]
        if label == "O":
            current = None
        elif current is not None and label[0] == "I" and (current["type"] == label[2:]):
            current["end"] = word[1]
        else:
            current = {"start": word[0], "end": word[1], "type": label[2:]}
            out.append(current)
    return out


def teach(logits, targets, labels, torch):
    """Compute masked temperature-two teacher distillation loss."""
    log = torch.nn.functional.log_softmax(logits.float() / 2, dim=-1)
    want = torch.softmax(targets.detach().float() / 2, dim=-1)
    loss = torch.nn.functional.kl_div(log, want, reduction="none").sum(-1)
    return loss[labels != -100].mean() * 4


def score(rows, predictions, tags):
    """Measure exact spans and character masking for Deep selection."""
    if not len(rows) == len(predictions) == len(tags):
        raise ValueError("Prediction row count mismatch")
    counts = {kind: Counter() for kind in TYPES}
    correct, total, covered, gold, false = (0, 0, 0, 0, 0)
    char_tp = char_fp = char_fn = 0
    failures = []
    for row, found, labels in zip(rows, predictions, tags, strict=True):
        want = {(span["start"], span["end"], span["type"]) for span in row["spans"]}
        got = {(span["start"], span["end"], span["type"]) for span in found}
        for kind in TYPES:
            counts[kind]["tp"] += len({span for span in want & got if span[2] == kind})
            counts[kind]["fp"] += len({span for span in got - want if span[2] == kind})
            counts[kind]["fn"] += len({span for span in want - got if span[2] == kind})
        if len(row["tags"]) != len(labels) or len(row["words"]) != len(row["tags"]):
            raise ValueError("Token prediction length mismatch")
        for word, left, right in zip(row["words"], row["tags"], labels, strict=True):
            if not ignored(row, *word):
                correct += int(int(left) == int(right))
                total += 1
        expected = bytearray(len(row["text"]))
        predicted = bytearray(len(row["text"]))
        for start, end, unused in want:
            expected[start:end] = b"\x01" * (end - start)
        for start, end, unused in got:
            predicted[start:end] = b"\x01" * (end - start)
        for position in range(len(expected)):
            char_tp += expected[position] and predicted[position]
            char_fp += not expected[position] and predicted[position]
            char_fn += expected[position] and (not predicted[position])
        for start, end, kind in want:
            cursor = start
            for span in found:
                if span["start"] <= cursor < span["end"]:
                    cursor = span["end"]
            covered += cursor >= end
            gold += 1
        false += len(got - want)
        if want != got and len(failures) < 10:
            failures.append({"text": row["text"], "gold": row["spans"], "found": found})

    def metric(count):
        """Compute precision, recall, F1 and F2 from counts."""
        true, false, missed = (count["tp"], count["fp"], count["fn"])
        precision, recall = (true / max(1, true + false), true / max(1, true + missed))
        return {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
            "support": true + missed,
            "tp": true,
            "fp": false,
            "fn": missed,
        }

    classes = {kind: metric(count) for kind, count in counts.items()}
    result = metric(sum(counts.values(), Counter()))
    precision, recall = (
        char_tp / max(1, char_tp + char_fp),
        char_tp / max(1, char_tp + char_fn),
    )
    result.update(
        masking_precision=precision,
        masking_recall=recall,
        masking_f1=2 * precision * recall / max(1e-12, precision + recall),
        masking_f2=5 * precision * recall / max(1e-12, 4 * precision + recall),
        accuracy=correct / max(1, total),
        coverage=covered / max(1, gold),
        macro=sum((item["f1"] for item in classes.values())) / len(TYPES),
        classes=classes,
        rows=len(rows),
        false=false,
        failures=failures,
    )
    return result


def predict(rows, run, collate, numpy):
    """Merge window argmax labels using the most central token."""
    entries = [(doc, *entry) for doc, row in enumerate(rows) for entry in windows(row)]
    labels = [numpy.zeros(len(row["words"]), dtype=numpy.int32) for row in rows]
    scores = [numpy.full(len(row["words"]), -1.0) for row in rows]
    start = time.perf_counter()
    for offset in range(0, len(entries), 32):
        batch = entries[offset : offset + 32]
        data = collate(
            [
                {key: value for key, value in entry[-1].items() if key != "labels"}
                for entry in batch
            ]
        )
        logits = run(data)
        for position, (doc, lower, upper, left, right, unused) in enumerate(batch):
            row = rows[doc]
            for word in range(lower, upper):
                token = row["heads"][word]
                central = min(
                    math.inf if left == 0 else token - left,
                    math.inf if right == len(row["ids"]) else right - 1 - token,
                )
                if central > scores[doc][word]:
                    labels[doc][word] = logits[position, token - left + 1].argmax()
                    scores[doc][word] = central
    elapsed = time.perf_counter() - start
    found = [spans(row["words"], tags) for row, tags in zip(rows, labels, strict=True)]
    result = score(rows, found, labels)
    result.update(seconds=elapsed, windows=len(entries))
    return result


def pack(model, tokenizer, path, torch):
    """Export and check the selected dynamic INT8 ONNX format."""
    import numpy
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    path.mkdir(exist_ok=True)
    model.cpu().eval()
    sample = tokenizer("My name is Priya Sharma.", return_tensors="pt")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    model.config.return_dict = False
    torch.onnx.export(
        model,
        tuple((sample[name] for name in names)),
        str(path / "float.onnx"),
        input_names=names,
        output_names=["logits"],
        dynamic_axes={name: {0: "batch", 1: "seq"} for name in names + ["logits"]},
        opset_version=17,
        dynamo=False,
    )
    model.config.return_dict = True
    quantize_dynamic(
        str(path / "float.onnx"),
        str(path / "model.onnx"),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gather"],
        per_channel=True,
    )
    graph = onnx.load(str(path / "model.onnx"))
    onnx.checker.check_model(graph)
    assert not any(
        (
            tensor.data_type == onnx.TensorProto.FLOAT
            and math.prod(tensor.dims) > 1000000
            for tensor in graph.graph.initializer
        )
    ), "Large float weights remain"
    tokenizer.backend_tokenizer.save(str(path / "tokenizer.json"))
    (path / "labels.json").write_text(json.dumps(LABELS))
    fixtures, skipped = prepare(
        [
            {"text": text, "spans": []}
            for text in (
                "Hi, I'm Priya Sharma from Infosys.",
                "Contact 中文John Smith.",
                "Anne\tO'Neill",
                "😀 José lives nearby.",
                "I\x00van works here.",
            )
        ],
        tokenizer,
    )
    assert not skipped
    (path / "tokens.json").write_text(
        json.dumps(
            [{key: row[key] for key in ("text", "ids", "words")} for row in fixtures]
        )
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path / "model.onnx"), options, providers=["CPUExecutionProvider"]
    )
    logits = session.run(None, {name: sample[name].numpy() for name in names})[0]
    assert (
        logits.shape == (1, sample["input_ids"].shape[1], len(LABELS))
        and numpy.isfinite(logits).all()
    )
    return session
