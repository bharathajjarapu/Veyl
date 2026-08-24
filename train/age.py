"""Train and select the original nonnegative AGE residual on frozen Deep."""

import copy
import hashlib
import json
import math
import random
import shutil
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from .common import LABELS, TYPES, schema, windows, prepare, spans, pack

NEGATIVE = ("AGE", "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "TAX_ID")
PROTOCOL = {
    "name": "age-residual-v1",
    "seed": 7,
    "steps": 1200,
    "batch_size": 32,
    "sequence_length": 512,
    "warmup_steps": 60,
    "learning_rate": 0.003,
    "weight_decay": 0.0,
    "clip_norm": 1.0,
    "source_slots": {
        "nv.train": 27648,
        "open.train": 9600,
        "luna.train": 768,
        "benign.train": 384,
    },
    "benign_rows": 128,
    "max_benign_draws_per_row": 3,
    "calibration_head_bias_delta": [-2, -1, 0, 1, 2],
    "fusion_manifest_sha256": "6b67c4b28e30c6e55bc6d797073d3f8d9a88751e2d97281763f5f3db6bde5576",
    "benign_train_sha256": "5f44743b4ed7090308c3563149030efb3fa210916431e0ff4c7f5fe5c8a5d209",
    "tagger_report_sha256": "283a547319c99cc0f9947e2bf072823944f6b49dfc7d6fd0c4e10558de5fec44",
    "shared_job_helper_sha256": "4143ef7391b9f14469d91fee7b6d3133776d2f1b3c44e5e288855a90ab22e190",
    "shared_tune_helper_sha256": "bfa64d96626c2aeced3253ae9deeb5a9315dc905e755732dbcf7166db29e070f",
    "selected_tokenizer_sha256": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "labels_sha256": "b945b66d77111228f7aae42cf85b4547e2f1b3a01a3c723b16e50e03db0a2253",
    "tokens_sha256": "9f9ad94244761ff241c7afeabc592fce3289b1a0efbae0da9b47c5ebbd8cb69a",
    "base_onnx_sha256": "272ac7467d46ff8e2e72f895df03a294ca824a5a9c0bb0d0748ca4ed244ebf52",
    "tagger_raw_onnx_sha256": "add6708b4b711bf114c64046fe64c81fb6ae1f49db21a660ba811243a2a190f0",
    "types_sha256": "3cafa0eb7c70c46d42acc25c85a9562f7f6e07d117fa3507a99cf3b404034c87",
    "input_files": [
        "job.json",
        "types.json",
        "base.onnx",
        "nv.train.jsonl",
        "nv.valid.jsonl",
        "open.train.jsonl",
        "open.valid.jsonl",
        "luna.train.jsonl",
        "haiku.valid.jsonl",
        "benign.train.jsonl",
    ],
    "negative_coverage": {
        "open.train": [
            "AGE",
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "TAX_ID",
        ],
        "fully_negative": "all 28 types",
    },
    "selection_gates": {
        "nv_char_f2_drop_max": 0.003,
        "nv_char_fpr_increase_max": 0.001,
        "open_char_f2_drop_max": 0.003,
        "haiku_benign_fp_chars_max": "baseline",
        "haiku_strict_age_fp_max": "baseline",
        "nv_datetime_to_age_overlap_chars_max": "baseline",
        "date_off_char_f2_drop_max": 0.003,
        "date_off_char_fpr_increase_max": 0.001,
        "open_strict_age_recall_gain_min": 0.1,
        "macro_source_char_f2_must_exceed_baseline": True,
    },
    "benchmark_or_guard_read": False,
}
BEGIN = LABELS.index("B-AGE")
INSIDE = LABELS.index("I-AGE")
assert (BEGIN, INSIDE) == (17, 18)


def corpus(folder, name):
    """Read the shared NVIDIA snapshots from Broad."""
    if name in ("nv.train.jsonl", "nv.valid.jsonl"):
        return folder.parent / "broad" / name[3:]
    return folder / name


def metric(true, false, missed):
    """Compute precision, recall, F1 and F2 from counts."""
    precision, recall = (true / max(1, true + false), true / max(1, true + missed))
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "f2": 5 * precision * recall / max(1e-12, 4 * precision + recall),
        "tp": int(true),
        "fp": int(false),
        "fn": int(missed),
    }


def gaps(mask):
    """Find contiguous unmarked character ranges."""
    out = []
    index = 0
    while index < len(mask):
        if mask[index]:
            index += 1
            continue
        position = index + 1
        while position < len(mask) and (not mask[position]):
            position += 1
        out.append((index, position))
        index = position
    return out


def negative(row):
    """Read the classes that a source annotates exhaustively."""
    if "negative" in row:
        return set(row["negative"])
    if row.get("source") == "openpii":
        return set(NEGATIVE)
    if row.get("source") in ("nvidia", "nv", "nvidia_train") or row.get(
        "fully_negative"
    ):
        return set(TYPES)
    return set()


def alternatives(negative):
    """Allow O and classes outside a row’s negative coverage."""
    known_negative = set(negative)
    allowed = [True]
    for kind in TYPES:
        allowed.extend([kind not in known_negative, kind not in known_negative])
    return allowed


def measure(rows, found, tags, types):
    """Character span metrics over only annotated types and nonignored characters."""
    totals = Counter()
    char_tp = char_fp = char_fn = char_neg = 0
    benign_fp = age_hit = password_hit = age_total = password_total = 0
    token_count = 0
    for row, pred, predtags in zip(rows, found, tags, strict=True):
        coverage = set(row.get("coverage", TYPES))
        ignored_ranges = list(row.get("ignore", []))
        truth = {
            (span["start"], span["end"], span["type"])
            for span in row["spans"]
            if span["type"] in coverage
        }
        guess = {
            (span["start"], span["end"], span["type"])
            for span in pred
            if span["type"] in coverage
        }
        strict_guess = {
            span
            for span in guess
            if not any(
                (left < span[1] and span[0] < right for left, right in ignored_ranges)
            )
        }
        for left, right, kind in truth:
            totals[kind, "tp"] += (left, right, kind) in strict_guess
            totals[kind, "fn"] += (left, right, kind) not in strict_guess
        for left, right, kind in strict_guess - truth:
            totals[kind, "fp"] += 1
        positive = bytearray(len(row["text"]))
        predicted = bytearray(len(row["text"]))
        ignored_mask = bytearray(len(row["text"]))
        for left, right, kind in truth:
            positive[left:right] = b"\x01" * (right - left)
        for left, right, kind in guess:
            predicted[left:right] = b"\x01" * (right - left)
        for left, right in ignored_ranges:
            ignored_mask[left:right] = b"\x01" * (right - left)
        for index in range(len(positive)):
            if ignored_mask[index]:
                continue
            char_tp += positive[index] and predicted[index]
            char_fn += positive[index] and (not predicted[index])
            char_fp += not positive[index] and predicted[index]
            char_neg += not positive[index]
        for left, right, kind in truth:
            covered = all((predicted[index] for index in range(left, right)))
            if kind == "AGE":
                age_total += 1
                age_hit += covered
            if kind == "PASSWORD":
                password_total += 1
                password_hit += covered
        if row.get("source") == "haiku" and row.get("fully_negative"):
            benign_fp += sum(
                (
                    1
                    for index, item in enumerate(predicted)
                    if item and (not ignored_mask[index])
                )
            )
        token_count += len(row["words"])
    per_type = {
        kind: metric(totals[kind, "tp"], totals[kind, "fp"], totals[kind, "fn"])
        for kind in types
        if kind in set().union(*(set(record.get("coverage", TYPES)) for record in rows))
    }
    micro = metric(char_tp, char_fp, char_fn)
    macro = sum((value["f2"] for value in per_type.values())) / max(1, len(per_type))
    return {
        "char_f2": micro["f2"],
        "char_precision": micro["precision"],
        "char_recall": micro["recall"],
        "char_fpr": char_fp / max(1, char_neg),
        "macro_type_f2": macro,
        "types": per_type,
        "benign_fp_chars": int(benign_fp),
        "age_span_union": int(age_hit),
        "age_span_total": int(age_total),
        "password_span_union": int(password_hit),
        "password_span_total": int(password_total),
        "word_tokens": int(token_count),
        "rows": len(rows),
    }


def objective(logits, labels, coverage, weights, torch):
    """Weighted positive CE; O marginal keeps alternatives for incompletely annotated types."""
    import torch.nn.functional as functional

    scores = logits.float()
    target = labels.long()
    valid = target.ne(-100)
    if not valid.any():
        return scores.sum() * 0
    logprob = functional.log_softmax(scores, dim=-1)
    positive = valid & target.ne(0)
    losses = torch.zeros_like(target, dtype=torch.float32)
    if positive.any():
        positive_labels = target[positive]
        positive_logp = logprob[positive]
        losses[positive] = (
            -positive_logp.gather(1, positive_labels.unsqueeze(1)).squeeze(1)
            * weights[positive_labels]
        )
    negative = valid & target.eq(0)
    if negative.any():
        allow = coverage[:, None, :].expand_as(scores).clone()
        allow[:, :, 0] = True
        mass = torch.logsumexp(logprob.masked_fill(~allow, -10000.0), dim=-1)
        losses[negative] = -mass[negative]
    return losses[valid].mean()


def finite(loss, torch):
    """Reject a nonfinite training objective."""
    if not bool(torch.isfinite(loss).all().item()):
        raise FloatingPointError("Non-finite training loss")
    return loss


def losscheck(torch=None):
    """Check coverage masks, ignored labels and marginal O loss."""
    assert gaps(bytearray([0, 0, 1, 0])) == [(0, 2), (3, 4)]
    assert negative(
        {"source": "openpii", "coverage": ["PERSON", "SENSITIVE_ATTRIBUTE"]}
    ) == set(NEGATIVE)
    assert (
        negative({"source": "haiku", "fully_negative": False, "negative": []}) == set()
    )
    assert alternatives(TYPES) == [True] + [False] * (len(LABELS) - 1)
    assert alternatives([]) == [True] * len(LABELS)
    open_allowed = alternatives(NEGATIVE)
    assert (
        not open_allowed[LABELS.index("B-PERSON")]
        and open_allowed[LABELS.index("B-LOCATION")]
    )
    row = {
        "text": "abcXYZ",
        "spans": [{"start": 0, "end": 3, "type": "PERSON"}],
        "coverage": ["PERSON"],
        "ignore": [[3, 6]],
        "words": [(0, 3), (3, 6)],
        "tags": [1, 0],
    }
    ignored_score = measure(
        [row], [[{"start": 0, "end": 6, "type": "PERSON"}]], [[1, 1]], ["PERSON"]
    )
    assert (
        ignored_score["types"]["PERSON"]["fp"] == 0 and ignored_score["char_f2"] == 1.0
    )
    title_row = {
        "text": "Title Jane",
        "spans": [{"start": 6, "end": 10, "type": "PERSON"}],
        "coverage": ["PERSON"],
        "ignore": [[0, 6]],
        "words": [(0, 5), (6, 10)],
        "tags": [0, 1],
    }
    title_score = measure(
        [title_row], [[{"start": 6, "end": 10, "type": "PERSON"}]], [[0, 1]], ["PERSON"]
    )
    assert title_score["char_f2"] == 1.0
    if torch is not None:
        logits = torch.tensor(
            [[[3.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]]],
            requires_grad=True,
        )
        labels = torch.tensor([[0, 1, -100]])
        coverage = torch.tensor([[True, False, True, False]])
        weights = torch.tensor([1.0, 2.0, 1.0, 1.0])
        loss = objective(logits, labels, coverage, weights, torch)
        loss.backward()
        assert bool(torch.isfinite(loss).all().item()) and logits.grad[0, 2].eq(0).all()
        try:
            finite(torch.tensor(float("nan")), torch)
        except FloatingPointError:
            pass
        else:
            raise AssertionError("Non-finite loss was not rejected")
        logprob = torch.log_softmax(logits.detach()[0].float(), dim=-1)
        expected = (
            -torch.logsumexp(logprob[0, [0, 2]], dim=0) + 2 * -logprob[1, 1]
        ) / 2
        assert torch.allclose(loss.detach(), expected)
        assert (
            logits.grad[0, 0, 2] < 0
            and logits.grad[0, 0, 1] > 0
            and (logits.grad[0, 1, 1] < 0)
        )
        rowlogits = torch.tensor([[[0.0, 1.0, 2.0]]], requires_grad=True)
        rowlabels = torch.tensor([[0]])
        only = objective(
            rowlogits,
            rowlabels,
            torch.tensor([[True, False, False]]),
            torch.ones(3),
            torch,
        )
        expectedloss = -torch.log_softmax(rowlogits.float(), dim=-1)[0, 0, 0]
        assert torch.allclose(only, expectedloss)
        unknown = objective(
            rowlogits,
            rowlabels,
            torch.tensor([[True, True, True]]),
            torch.ones(3),
            torch,
        )
        assert abs(float(unknown.detach())) < 1e-06
    return True


def head(torch, base):
    """Add a frozen-body Softplus residual to the two AGE logits."""

    class Residual(torch.nn.Module):
        def __init__(self, model):
            """Initialize the frozen body and zero-weight AGE residual."""
            super().__init__()
            self.base = model
            self.head = torch.nn.Linear(model.config.hidden_size, 2)
            torch.nn.init.zeros_(self.head.weight)
            torch.nn.init.constant_(self.head.bias, -10.0)
            for value in self.base.parameters():
                value.requires_grad_(False)

        @property
        def config(self):
            """Expose the frozen classifier configuration."""
            return self.base.config

        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            """Compute the model outputs for one token batch."""
            hidden = self.base.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=False,
            )[0]
            logits = self.base.classifier(self.base.dropout(hidden))
            delta = torch.nn.functional.softplus(self.head(hidden))
            logits = torch.cat(
                (
                    logits[..., :BEGIN],
                    logits[..., BEGIN : INSIDE + 1] + delta,
                    logits[..., INSIDE + 1 :],
                ),
                dim=-1,
            )
            return (
                SimpleNamespace(logits=logits) if self.config.return_dict else (logits,)
            )

    return Residual(base)


def headcheck(torch):
    """Check the 258 trainable parameters and protected logits."""

    class Config:
        hidden_size = 128
        return_dict = True

    class Encoder(torch.nn.Module):
        def forward(
            self, input_ids, attention_mask=None, token_type_ids=None, return_dict=False
        ):
            """Compute the model outputs for one token batch."""
            item = torch.nn.functional.one_hot(input_ids.remainder(128), 128).float()
            return (item,)

    class Base(torch.nn.Module):
        def __init__(self):
            """Initialize the frozen body and zero-weight AGE residual."""
            super().__init__()
            self.config = Config()
            self.bert = Encoder()
            self.dropout = torch.nn.Identity()
            self.classifier = torch.nn.Linear(128, 57)

        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            """Compute the model outputs for one token batch."""
            hidden = self.bert(input_ids, attention_mask, token_type_ids, False)[0]
            return SimpleNamespace(logits=self.classifier(hidden))

    base = Base()
    model = head(torch, base)
    model.eval()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    assert (
        sum((parameter.numel() for parameter in trainable)) == 258
        and len(trainable) == 2
        and all(
            (
                left is right
                for left, right in zip(trainable, model.head.parameters(), strict=True)
            )
        )
    )
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    old = base(ids).logits
    new = model(ids).logits
    assert new.shape == (1, 3, 57) and torch.isfinite(new).all()
    assert torch.equal(old[..., :BEGIN], new[..., :BEGIN]) and torch.equal(
        old[..., INSIDE + 1 :], new[..., INSIDE + 1 :]
    )
    assert bool((new[..., BEGIN : INSIDE + 1] >= old[..., BEGIN : INSIDE + 1]).all())
    model.config.return_dict = False
    assert isinstance(model(ids), tuple) and model(ids)[0].shape == (1, 3, 57)
    model.config.return_dict = True
    loss = torch.nn.functional.cross_entropy(
        model(ids).logits.reshape(-1, 57), torch.full((3,), BEGIN)
    )
    loss.backward()
    assert torch.isfinite(model.head.weight.grad).all() and model.head.bias.grad[0] < 0
    assert all((parameter.grad is None for parameter in base.parameters()))
    return True


def smoke(session, cache, numpy, atol=0.0001, rtol=1e-05):
    """Check non-AGE logits stay fixed and AGE gains stay nonnegative."""
    checked = 0
    maximum = 0.0
    minimum_gain = math.inf
    keep = [index for index in range(len(LABELS)) if index not in (BEGIN, INSIDE)]
    for inputs, expected in cache:
        actual = session.run(None, inputs)[0]
        if (
            actual.shape != expected.shape
            or not numpy.isfinite(actual).all()
            or (not numpy.isfinite(expected).all())
        ):
            return {
                "passed": False,
                "checked_windows": checked,
                "reason": "shape_or_finite",
            }
        error = numpy.abs(actual[:, :, keep] - expected[:, :, keep])
        maximum = max(maximum, float(error.max(initial=0.0)))
        gain = actual[:, :, BEGIN : INSIDE + 1] - expected[:, :, BEGIN : INSIDE + 1]
        minimum_gain = min(minimum_gain, float(gain.min()))
        checked += 1
        if not numpy.allclose(
            actual[:, :, keep], expected[:, :, keep], atol=atol, rtol=rtol
        ) or bool((gain < -atol).any()):
            return {
                "passed": False,
                "checked_windows": checked,
                "max_non_age_abs_error": maximum,
                "min_age_gain": minimum_gain,
            }
    return {
        "passed": checked >= 2,
        "checked_windows": checked,
        "max_non_age_abs_error": maximum,
        "min_age_gain": float(minimum_gain),
        "atol": float(atol),
        "rtol": float(rtol),
    }


def smokecheck(session, cache, numpy):
    """Apply the export check to the complete validation cache."""
    return smoke(session, cache, numpy)


def parity(session, cache, numpy, atol=0.001, rtol=0.0001):
    """Compare the initial residual export with calibrated Deep logits."""
    checked = 0
    maximum = 0.0
    for inputs, expected in cache:
        actual = session.run(None, inputs)[0]
        if (
            actual.shape != expected.shape
            or not numpy.isfinite(actual).all()
            or (not numpy.isfinite(expected).all())
        ):
            return {
                "passed": False,
                "checked_windows": checked,
                "reason": "shape_or_finite",
            }
        error = numpy.abs(actual - expected)
        maximum = max(maximum, float(error.max(initial=0.0)))
        checked += 1
        if not numpy.allclose(actual, expected, atol=atol, rtol=rtol):
            return {
                "passed": False,
                "checked_windows": checked,
                "max_abs_error": maximum,
                "atol": float(atol),
                "rtol": float(rtol),
            }
    return {
        "passed": checked >= 2,
        "checked_windows": checked,
        "max_abs_error": maximum,
        "atol": float(atol),
        "rtol": float(rtol),
    }


def exportcheck(numpy):
    """Check parity failures and singleton-window export smoke coverage."""
    cache = []
    for size in (3, 5, 8):
        values = numpy.arange(size * len(LABELS), dtype=numpy.float32).reshape(
            1, size, len(LABELS)
        )
        cache.append(({"input_ids": numpy.zeros((1, size), dtype=numpy.int64)}, values))

    class Session:
        def __init__(self, mode):
            """Select a synthetic export-check mode."""
            self.mode = mode
            self.seen = []

        def run(self, outputs, data):
            """Return synthetic logits for the export check."""
            size = data["input_ids"].shape[1]
            self.seen.append((1, size))
            value = next(
                (item for item in cache if item[0]["input_ids"].shape[1] == size)
            )[1].copy()
            if self.mode == "bad":
                value[0, 0, 0] += 1
            if self.mode == "nan":
                value[0, 0, 0] = numpy.nan
            value[:, :, BEGIN : INSIDE + 1] += 0.25
            return [value]

    good = Session("good")
    assert smokecheck(good, cache, numpy)["passed"] and good.seen == [
        (1, 3),
        (1, 5),
        (1, 8),
    ]
    two = Session("good")
    assert smokecheck(two, cache[:2], numpy)["checked_windows"] == 2 and two.seen == [
        (1, 3),
        (1, 5),
    ]
    assert not smokecheck(Session("bad"), cache, numpy)["passed"]
    assert not smokecheck(Session("nan"), cache, numpy)["passed"]

    class ParitySession:
        def __init__(self, bad=False):
            """Select whether the parity fixture changes one logit."""
            self.bad = bad

        def run(self, outputs, data):
            """Return synthetic logits for the export check."""
            size = data["input_ids"].shape[1]
            value = next(
                (item for item in cache if item[0]["input_ids"].shape[1] == size)
            )[1].copy()
            if self.bad:
                value[0, 0, 0] += 1
            return [value]

    assert parity(ParitySession(), cache, numpy)["passed"]
    assert not parity(ParitySession(True), cache, numpy)["passed"]
    return True


def annotate(rows, source):
    """Apply source-specific negative coverage and ignored characters."""
    for row in rows:
        if source.startswith("nv.") or source == "benign.train":
            row["negative"] = list(TYPES)
        elif source.startswith("open."):
            row["negative"] = list(NEGATIVE)
        else:
            row["negative"] = (
                list(TYPES) if row.get("fully_negative") else row.get("negative", [])
            )
        if source == "benign.train":
            if row.get("spans") or row.get("ignore"):
                raise ValueError(
                    "Benign residual examples must be unannotated all-negative rows"
                )
            row["source"] = "benign"
            row["fully_negative"] = True
            row["coverage"] = list(TYPES)
        elif source.startswith("nv."):
            row["source"] = "nv"
            row["coverage"] = list(TYPES)
        elif source.startswith("open."):
            row["source"] = "openpii"
        elif source.startswith("luna."):
            row["source"] = "luna"
        elif source.startswith("haiku."):
            row["source"] = "haiku"
        if source.startswith(("luna.", "haiku.")) and (
            not row.get("fully_negative", False)
        ):
            marked = bytearray(len(row["text"]))
            for span in row.get("spans", []):
                marked[span["start"] : span["end"]] = b"\x01" * (
                    span["end"] - span["start"]
                )
            row["ignore"] = list(row.get("ignore", [])) + [
                [left, right] for left, right in gaps(marked)
            ]
    return rows


def entries(rows):
    """Keep row identity and negative coverage on each token window."""
    out = []
    for row in rows:
        for entry in windows(row):
            item = copy.deepcopy(entry[-1])
            item["_negative"] = sorted(negative(row))
            item["_source"] = row.get("source")
            item["_row_id"] = row.get("id", row.get("uid"))
            out.append(item)
    return out


def evaluate(rows, found, tags, base_tags=None):
    """Measure AGE recall, false positives and date confusion."""
    metrics = measure(rows, found, tags, TYPES)
    gold = {
        (document, span["start"], span["end"])
        for document, row in enumerate(rows)
        for span in row["spans"]
        if span["type"] == "AGE"
    }
    guess = {
        (document, span["start"], span["end"])
        for document, pred in enumerate(found)
        for span in pred
        if span["type"] == "AGE"
    }
    false = sum(
        (
            1
            for document, left, right in guess - gold
            if rows[document].get("fully_negative")
            and rows[document].get("source") in ("haiku", "benign")
        )
    )
    whitespace = sum(
        (
            1
            for document, pred in enumerate(found)
            for span in pred
            if span["type"] == "AGE"
            and any(
                (
                    character.isspace()
                    for character in rows[document]["text"][span["start"] : span["end"]]
                )
            )
        )
    )
    confusion = []
    confusion_chars = 0
    for row, pred in zip(rows, found, strict=True):
        dates = [span for span in row["spans"] if span["type"] == "DATE_TIME"]
        for age in (span for span in pred if span["type"] == "AGE"):
            overlap = sum(
                (
                    max(
                        0,
                        min(age["end"], date["end"]) - max(age["start"], date["start"]),
                    )
                    for date in dates
                )
            )
            if overlap:
                confusion.append(age)
                confusion_chars += overlap
    true = len(gold & guess)
    unmatched = len(guess - gold)
    missed = len(gold - guess)
    metrics["strict_age"] = {
        "tp": true,
        "fp": unmatched,
        "fn": missed,
        "precision": true / max(1, true + unmatched),
        "recall": true / max(1, true + missed),
        "fully_negative_haiku_age_fp_spans": false,
    }
    metrics["age_spans_containing_whitespace"] = int(whitespace)
    metrics["age_spans_overlapping_gold_datetime"] = {
        "spans": len(confusion),
        "overlap_chars": int(confusion_chars),
    }
    if base_tags is not None:
        metrics["label_changes_vs_frozen_deep"] = compare(base_tags, tags)
    return metrics


def dateoff(rows, found, tags):
    """Score character masking with DATE_TIME spans removed."""
    view = []
    predictions = []
    for row, pred in zip(rows, found, strict=True):
        item = dict(row)
        item["spans"] = [span for span in row["spans"] if span["type"] != "DATE_TIME"]
        view.append(item)
        predictions.append([span for span in pred if span["type"] != "DATE_TIME"])
    return measure(view, predictions, tags, TYPES)


def datecheck():
    """Check date-to-AGE theft remains a false positive."""
    row = {
        "text": "2034",
        "words": ["2034"],
        "spans": [{"start": 0, "end": 4, "type": "DATE_TIME"}],
        "coverage": ["AGE", "DATE_TIME"],
        "ignore": [],
        "source": "nv",
        "fully_negative": False,
    }
    date = [{"start": 0, "end": 4, "type": "DATE_TIME"}]
    age = [{"start": 0, "end": 4, "type": "AGE"}]
    base = dateoff([row], [date], [[0]])
    stolen = dateoff([row], [age], [[0]])
    assert base["char_fpr"] == 0 and stolen["char_fpr"] == 1
    return True


def predict(rows, session, collate, numpy):
    """Merge window argmax labels using the most central token."""
    entries = [(doc, *entry) for doc, row in enumerate(rows) for entry in windows(row)]
    labels = [numpy.zeros(len(row["words"]), dtype=numpy.int32) for row in rows]
    margins = [numpy.full(len(row["words"]), -1.0, dtype=numpy.float32) for row in rows]
    for doc, lower, upper, left, right, item in entries:
        batch = collate(
            [{key: value for key, value in item.items() if key != "labels"}]
        )
        inputs = {key: value.numpy() for key, value in batch.items()}
        logits = session.run(None, inputs)[0]
        if (
            logits.shape != (1, batch["input_ids"].shape[1], len(LABELS))
            or not numpy.isfinite(logits).all()
        ):
            raise ValueError("Invalid AGE ONNX logits")
        row = rows[doc]
        for word in range(lower, upper):
            token = row["heads"][word]
            center = min(
                math.inf if left == 0 else token - left,
                math.inf if right == len(row["ids"]) else right - 1 - token,
            )
            if center > margins[doc][word]:
                labels[doc][word] = int(logits[0, token - left + 1].argmax())
                margins[doc][word] = center
    found = [spans(row["words"], tag) for row, tag in zip(rows, labels, strict=True)]
    return (evaluate(rows, found, labels), found, labels)


def compare(base, new):
    """Count label changes against frozen Deep predictions."""
    total = protected = to_age = 0
    for left, right in zip(base, new, strict=True):
        for item, target in zip(left, right, strict=True):
            total += item != target
            if item not in (BEGIN, INSIDE):
                protected += item != target
                to_age += target in (BEGIN, INSIDE)
    return {
        "total_argmax_flips": int(total),
        "protected_non_age_to_age_flips": int(to_age),
        "other_non_age_flips": int(protected - to_age),
    }


def eligible(metrics, baseline, baseline_macro):
    """Apply every original validation gate to a checkpoint."""
    return bool(
        metrics["nv.valid"]["char_f2"] >= baseline["nv.valid"]["char_f2"] - 0.003
        and metrics["nv.valid"]["char_fpr"] <= baseline["nv.valid"]["char_fpr"] + 0.001
        and (
            metrics["open.valid"]["char_f2"]
            >= baseline["open.valid"]["char_f2"] - 0.003
        )
        and (
            metrics["haiku.valid"]["benign_fp_chars"]
            <= baseline["haiku.valid"]["benign_fp_chars"]
        )
        and (
            metrics["haiku.valid"]["strict_age"]["fully_negative_haiku_age_fp_spans"]
            <= baseline["haiku.valid"]["strict_age"][
                "fully_negative_haiku_age_fp_spans"
            ]
        )
        and (
            metrics["nv.valid"]["age_spans_overlapping_gold_datetime"]["overlap_chars"]
            <= baseline["nv.valid"]["age_spans_overlapping_gold_datetime"][
                "overlap_chars"
            ]
        )
        and all(
            (
                metrics[split]["date_off"]["char_f2"]
                >= baseline[split]["date_off"]["char_f2"] - 0.003
                and metrics[split]["date_off"]["char_fpr"]
                <= baseline[split]["date_off"]["char_fpr"] + 0.001
                for split in ("nv.valid", "open.valid", "haiku.valid")
            )
        )
        and (
            metrics["open.valid"]["strict_age"]["recall"]
            >= baseline["open.valid"]["strict_age"]["recall"] + 0.1
        )
        and (
            sum(
                (
                    metrics[split]["char_f2"]
                    for split in ("nv.valid", "open.valid", "haiku.valid")
                )
            )
            / 3
            > baseline_macro
        )
    )


def audit(onnx, source_path, target_path):
    """Verify that quantized export preserves all frozen base tensors."""
    from collections import Counter
    from onnx import numpy_helper

    def read(path):
        """Fingerprint graph initializers and classifier biases."""
        graph = onnx.load(str(path))
        counts = Counter()
        biases = []
        names = Counter()
        for tensor in graph.graph.initializer:
            values = numpy_helper.to_array(tensor)
            digest = hashlib.sha256(values.tobytes()).hexdigest()
            key = (
                int(tensor.data_type),
                tuple((int(item) for item in tensor.dims)),
                digest,
            )
            counts[key] += 1
            name = tensor.name.removeprefix("base.")
            names[name] += 1
            if "classifier" in name.lower() and "bias" in name.lower():
                biases.append((name, key))
        return (counts, biases, names)

    original, source_bias, source_names = read(source_path)
    exported, target_bias, target_names = read(target_path)
    missing = original - exported
    sourcebias = Counter((key for unused, key in source_bias))
    targetbias = Counter((key for unused, key in target_bias))
    duplicates = {
        "source": [
            number for number, character in source_names.items() if character > 1
        ],
        "target": [
            number for number, character in target_names.items() if character > 1
        ],
    }
    return {
        "source_initializers": int(sum(original.values())),
        "target_initializers": int(sum(exported.values())),
        "all_source_tensors_preserved": not missing,
        "missing_tensor_fingerprints": [
            {
                "dtype": key[0],
                "dims": list(key[1]),
                "sha256": key[2],
                "count": int(number),
            }
            for key, number in missing.items()
        ],
        "source_classifier_biases": [name for name, unused in source_bias],
        "target_classifier_biases": [name for name, unused in target_bias],
        "classifier_bias_tensor_preserved": bool(
            source_bias and (not sourcebias - targetbias)
        ),
        "duplicate_normalized_names": duplicates,
    }


def restore(out, source, deep, protocol):
    """Emit the pinned Deep graph when no AGE candidate qualifies."""
    target = out / "model"
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    shutil.copy2(source / "base.onnx", target / "model.onnx")
    for name in ("tokenizer.json", "labels.json", "tokens.json"):
        shutil.copy2(deep / name, target / name)
    hashes = {
        name: hashlib.sha256((target / name).read_bytes()).hexdigest()
        for name in ("model.onnx", "tokenizer.json", "labels.json", "tokens.json")
    }
    valid = (
        hashes["tokenizer.json"] == protocol["selected_tokenizer_sha256"]
        and hashes["labels.json"] == protocol["labels_sha256"]
        and (hashes["tokens.json"] == protocol["tokens_sha256"])
    )
    if not valid:
        raise ValueError("Original Deep fallback assets failed pinned hashes")
    return {
        "model_sha256": hashes["model.onnx"],
        "model_bytes": (target / "model.onnx").stat().st_size,
        "asset_sha256": hashes,
        "assets_match_pins": bool(valid),
    }


def main():
    """Run this fixed training stage into a new output directory."""
    import numpy
    import torch

    assert torch.cuda.is_available(), "CUDA GPU unavailable; refusing CPU training"
    print({"stage": "start", "gpu": torch.cuda.get_device_name(0)}, flush=True)
    if (
        not isinstance(PROTOCOL["fusion_manifest_sha256"], str)
        or len(PROTOCOL["fusion_manifest_sha256"]) != 64
    ):
        raise ValueError("AGE Fusion v3 manifest SHA must be pinned before launch")
    headcheck(torch)
    exportcheck(numpy)
    datecheck()
    losscheck(torch)
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        get_linear_schedule_with_warmup,
    )
    from tokenizers import Tokenizer as BackendTokenizer

    random.seed(7)
    numpy.random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.set_num_threads(2)
    home = Path(__file__).resolve().parents[1]
    source = home / "data" / "tune"
    root = home / "runs" / "age"
    root.mkdir(parents=True, exist_ok=False)
    manifest = source / "data.json"
    meta = json.loads(manifest.read_text())
    if (
        meta.get("protocol") != PROTOCOL["name"]
        or hashlib.sha256(manifest.read_bytes()).hexdigest()
        != PROTOCOL["fusion_manifest_sha256"]
    ):
        raise ValueError("Fusion v3 manifest SHA mismatch")
    hashes = meta.get("input_hashes", {})
    names = tuple(PROTOCOL["input_files"])
    if any(
        (name not in hashes or not corpus(source, name).is_file() for name in names)
    ):
        raise ValueError("Fusion v3 required inputs missing")
    if hashes.get("benign.train.jsonl") != PROTOCOL["benign_train_sha256"]:
        raise ValueError("Audited benign input SHA mismatch")
    for name in names:
        if (
            Path(name).name != name
            or hashlib.sha256(corpus(source, name).read_bytes()).hexdigest()
            != hashes[name]
        ):
            raise ValueError("Fusion v3 input hash mismatch: " + name)
    if (
        hashlib.sha256((source / "types.json").read_bytes()).hexdigest()
        != PROTOCOL["types_sha256"]
    ):
        raise ValueError("AGE taxonomy differs from frozen 28-type schema")
    schema(json.loads((source / "types.json").read_text()))
    settings = json.loads((source / "job.json").read_text())
    tokens = settings.get("tokenizer", {})
    if tokens.get("base") != "google/bert_uncased_L-6_H-128_A-2":
        raise ValueError("Fusion tokenizer base mismatch")
    revision = tokens.get("revision")
    if revision != "cc3ddb10622cdf031e6c96cf314284cd788bc24b":
        raise ValueError("Fusion Deep revision mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        tokens["base"], use_fast=True, revision=revision
    )
    if len(tokenizer.get_vocab()) != 30522:
        raise ValueError("Deep tokenizer vocabulary mismatch")
    raw = {
        name: [
            json.loads(line)
            for line in corpus(source, name + ".jsonl").read_text().splitlines()
            if line
        ]
        for name in (
            "nv.train",
            "nv.valid",
            "open.train",
            "open.valid",
            "luna.train",
            "haiku.valid",
            "benign.train",
        )
    }
    for name, rows in raw.items():
        annotate(rows, name)
    if (
        len(raw["benign.train"]) != 128
        or len({record.get("id", record.get("uid")) for record in raw["benign.train"]})
        != 128
    ):
        raise ValueError("Expected 128 unique audited benign examples")
    data = {}
    audits = {}
    for name, rows in raw.items():
        data[name], audits[name] = prepare(rows, tokenizer)
    print(
        {
            "stage": "source_prepared",
            "rows": {key: len(value) for key, value in data.items()},
            "audit": audits,
        },
        flush=True,
    )
    if len(LABELS) != 57 or LABELS[BEGIN] != "B-AGE" or LABELS[INSIDE] != "I-AGE":
        raise ValueError("AGE BIO label positions mismatch")
    expected = meta.get("tagger_report_sha256")
    if expected != PROTOCOL["tagger_report_sha256"]:
        raise ValueError("Fusion v3 Tagger report SHA mismatch")
    provenance = home / "local" / "reports" / "tagger.json"
    if hashlib.sha256(provenance.read_bytes()).hexdigest() != expected:
        raise ValueError("Tagger report SHA mismatch")
    deep = home / "weights"
    checkpoint = deep / "deep"
    saved = json.loads(provenance.read_text())
    metadata = saved["models"]["deep"]
    if (
        metadata.get("base") != tokens["base"]
        or metadata.get("revision") != revision
        or saved.get("hashes", {}).get("types.json") != PROTOCOL["types_sha256"]
        or (saved.get("scope") != list(TYPES))
    ):
        raise ValueError("Tagger Deep metadata mismatch")
    if (
        hashlib.sha256((deep / "tokenizer.json").read_bytes()).hexdigest()
        != PROTOCOL["selected_tokenizer_sha256"]
    ):
        raise ValueError("Tagger Deep tokenizer SHA mismatch")
    if json.loads((deep / "labels.json").read_text()) != LABELS:
        raise ValueError("Tagger Deep label order mismatch")
    attached = BackendTokenizer.from_file(str(deep / "tokenizer.json"))
    if attached.get_vocab() != tokenizer.backend_tokenizer.get_vocab():
        raise ValueError("Attached tokenizer vocabulary mismatch")
    if (
        hashlib.sha256((source / "base.onnx").read_bytes()).hexdigest()
        != PROTOCOL["base_onnx_sha256"]
    ):
        raise ValueError("Deep calibrated ONNX SHA mismatch")
    if (
        hashlib.sha256((deep / "raw.onnx").read_bytes()).hexdigest()
        != PROTOCOL["tagger_raw_onnx_sha256"]
    ):
        raise ValueError("Tagger raw ONNX SHA mismatch")
    base = AutoModelForTokenClassification.from_pretrained(
        checkpoint, attn_implementation="eager"
    )
    if (
        base.config.num_labels != 57
        or base.config.hidden_size != 128
        or base.config.num_hidden_layers != 6
        or (base.config.num_attention_heads != 2)
    ):
        raise ValueError("Deep classifier dimensions mismatch")
    if {int(key): value for key, value in base.config.id2label.items()} != {
        index: name for index, name in enumerate(LABELS)
    }:
        raise ValueError("Deep checkpoint id2label order mismatch")
    original = base.classifier.bias.detach()[0].clone()
    base.classifier.bias.data[0] -= 1.0
    if not torch.equal(base.classifier.bias.detach()[0], original - 1.0):
        raise ValueError("Initial O-bias calibration was not applied exactly")
    model = head(torch, base).cuda().eval()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    assert sum((parameter.numel() for parameter in trainable)) == 258 and all(
        (not parameter.requires_grad for parameter in model.base.parameters())
    )
    collate = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=512)
    ort_collate = DataCollatorForTokenClassification(tokenizer)
    options = __import__("onnxruntime").SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    import onnxruntime as ort

    baseline_session = ort.InferenceSession(
        str(source / "base.onnx"), options, providers=["CPUExecutionProvider"]
    )
    splits = ("nv.valid", "open.valid", "haiku.valid")
    baseline = {}
    base_tags = {}
    for split in splits:
        measured, found, tags = predict(
            data[split], baseline_session, ort_collate, numpy
        )
        measured["date_off"] = dateoff(data[split], found, tags)
        baseline[split] = measured
        base_tags[split] = tags
    if (
        baseline["open.valid"]["strict_age"]["tp"]
        + baseline["open.valid"]["strict_age"]["fn"]
        == 0
    ):
        raise ValueError(
            "OpenPII validation has no AGE spans for the requested recall gain gate"
        )
    parity_cache = []
    for split in splits:
        for row in data[split][:1]:
            entry = windows(row)[0][-1]
            batch = ort_collate(
                [{key: value for key, value in entry.items() if key != "labels"}]
            )
            inputs = {key: value.numpy() for key, value in batch.items()}
            parity_cache.append((inputs, baseline_session.run(None, inputs)[0]))
    probe = root / "init"
    shutil.rmtree(probe, ignore_errors=True)
    pack(model, tokenizer, probe, torch)
    (probe / "float.onnx").unlink(missing_ok=True)
    import onnx

    probe_audit = audit(onnx, source / "base.onnx", probe / "model.onnx")
    probe_bytes = (probe / "model.onnx").stat().st_size
    probe_gate = {
        "bytes": int(probe_bytes),
        "within_10mb": probe_bytes <= 10000000,
        "audit": probe_audit,
    }
    (root / "age-init-probe.json").write_text(json.dumps(probe_gate, indent=2))
    if (
        not probe_audit["all_source_tensors_preserved"]
        or not probe_audit["classifier_bias_tensor_preserved"]
        or any(probe_audit["duplicate_normalized_names"].values())
        or (probe_bytes > 10000000)
    ):
        raise ValueError(
            "Initial residual export failed size/base-tensor preservation gate"
        )
    probe_session = ort.InferenceSession(
        str(probe / "model.onnx"), options, providers=["CPUExecutionProvider"]
    )
    base_parity = parity(probe_session, parity_cache, numpy)
    probe_gate["headless_base_logit_parity"] = base_parity
    (root / "age-init-probe.json").write_text(json.dumps(probe_gate, indent=2))
    if not base_parity["passed"]:
        raise ValueError("Initial export logits differ from pinned calibrated base")
    init = {}
    flips = {}
    for split in splits:
        unused, found_rows, tags = predict(
            data[split], probe_session, ort_collate, numpy
        )
        flips[split] = compare(base_tags[split], tags)
        init[split] = evaluate(data[split], found_rows, tags, base_tags[split])
    changes = sum((value["total_argmax_flips"] for value in flips.values()))
    probe_gate.update(
        {"initial_argmax_flip_count": int(changes), "initial_metrics": init}
    )
    (root / "age-init-probe.json").write_text(json.dumps(probe_gate, indent=2))
    if changes:
        raise ValueError("Initial AGE residual changed original validation argmax")
    shutil.rmtree(probe, ignore_errors=True)
    model.cuda().eval()
    report = {
        "protocol": PROTOCOL,
        "fusion_sha256": hashlib.sha256(
            (source / "data.json").read_bytes()
        ).hexdigest(),
        "input_sha256": dict(meta["input_hashes"]),
        "tagger": {
            "report_sha256": expected,
            "deep_revision": revision,
            "tokenizer_sha256": PROTOCOL["selected_tokenizer_sha256"],
            "initial_classifier_o_bias_delta": float(
                base.classifier.bias.detach()[0].cpu() - original.cpu()
            ),
            "raw_checkpoint_o_bias": float(original.cpu()),
            "calibrated_initial_o_bias": float(base.classifier.bias.detach()[0].cpu()),
        },
        "labels": {"count": len(LABELS), "age_b": BEGIN, "age_i": INSIDE},
        "parameters": {
            "total": int(sum((parameter.numel() for parameter in model.parameters()))),
            "trainable": int(sum((parameter.numel() for parameter in trainable))),
            "head": "Linear(128,2) + Softplus, added only to B-AGE/I-AGE",
        },
        "initial_export": {
            "bytes": int(probe_bytes),
            "base_initializer_audit": probe_audit,
            "headless_base_logit_parity": base_parity,
        },
        "baseline": baseline,
        "baseline_macro_source_char_f2": float(
            sum((baseline[split]["char_f2"] for split in splits)) / 3
        ),
        "initial": {
            "metrics": init,
            "argmax_flips_vs_frozen_body": flips,
            "argmax_flip_count": int(changes),
        },
        "history": [],
        "calibration": [],
        "ready": False,
        "provisional": True,
    }
    out = root / "age"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    (root / "report.json").write_text(json.dumps(report, indent=2))
    sets = {
        name: entries(data[name]) for name in ("nv.train", "open.train", "luna.train")
    }
    benign = entries(data["benign.train"])
    if any((not sets[key] for key in sets)) or len(benign) < 128:
        raise ValueError("Empty AGE training source")
    quotas = PROTOCOL["source_slots"]
    seq = [name for name, count in quotas.items() for unused in range(count)]
    assert len(seq) == 1200 * 32
    rng = random.Random(7)
    rng.shuffle(seq)
    # Keep exactly three exposures of each audited benign row.
    benignindices = []
    for row in data["benign.train"]:
        row_entries = [
            position
            for position, entry in enumerate(benign)
            if entry.get("_row_id") == row.get("id", row.get("uid"))
        ]
        if not row_entries:
            raise ValueError("Could not map benign row to prepared window")
        benignindices.extend([row_entries[0]] * 3)
    from collections import Counter

    exposure = Counter(benignindices)
    if (
        len(benignindices) != 384
        or len(exposure) != 128
        or any((number != 3 for number in exposure.values()))
    ):
        raise ValueError("Each of 128 benign rows must appear exactly three times")
    rng.shuffle(benignindices)
    benignplan = iter(benignindices)
    plan = []
    for step in range(1200):
        draw = []
        for slot, name in enumerate(seq[step * 32 : (step + 1) * 32]):
            if name == "benign.train":
                index = next(benignplan)
                origin = "benign.train"
                group = benign
            else:
                origin = name
                group = sets[name]
                index = random.Random(
                    7
                    + step * 32
                    + slot
                    + {"nv.train": 101, "open.train": 202, "luna.train": 303}[name]
                ).randrange(len(group))
            draw.append((origin, index))
        plan.append(draw)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.003, weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(optimizer, 60, 1200)
    scaler = torch.amp.GradScaler("cuda")
    weights = torch.ones(len(LABELS), device="cuda")
    history = []
    baseline_macro = float(sum((baseline[split]["char_f2"] for split in splits)) / 3)
    best = baseline_macro
    best_state = None
    best_step = None
    best_improved = False
    successful = skipped = 0
    start = time.time()
    group_by = {
        "nv.train": sets["nv.train"],
        "open.train": sets["open.train"],
        "luna.train": sets["luna.train"],
        "benign.train": benign,
    }
    for stepno, draw in enumerate(plan, 1):
        items = [group_by[origin][index] for origin, index in draw]
        batch = collate(
            [
                {key: value for key, value in item.items() if not key.startswith("_")}
                for item in items
            ]
        )
        allowed = torch.tensor(
            [alternatives(item["_negative"]) for item in items],
            dtype=torch.bool,
            device="cuda",
        )
        batch = {key: value.cuda() for key, value in batch.items()}
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(
                **{key: value for key, value in batch.items() if key != "labels"}
            ).logits
            loss = finite(
                objective(logits, batch["labels"], allowed, weights, torch), torch
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= before:
            scheduler.step()
            successful += 1
        else:
            skipped += 1
        if stepno % 100 == 0:
            (root / "report.json").write_text(json.dumps(report, indent=2))
            print(
                {
                    "stage": "train",
                    "step": stepno,
                    "successful_updates": successful,
                    "skipped_updates": skipped,
                    "loss": float(loss.detach().cpu()),
                    "lr": float(scheduler.get_last_lr()[0]),
                },
                flush=True,
            )
        if stepno % 300 == 0:
            model.eval()
            details = {}
            checkpoint_path = out / f"checkpoint-probe-{stepno}"
            shutil.rmtree(checkpoint_path, ignore_errors=True)
            pack(model, tokenizer, checkpoint_path, torch)
            check_session = ort.InferenceSession(
                str(checkpoint_path / "model.onnx"),
                options,
                providers=["CPUExecutionProvider"],
            )
            for split in splits:
                unused, found_rows, tags = predict(
                    data[split], check_session, ort_collate, numpy
                )
                details[split] = evaluate(
                    data[split], found_rows, tags, base_tags[split]
                )
                details[split]["date_off"] = dateoff(data[split], found_rows, tags)
            shutil.rmtree(checkpoint_path, ignore_errors=True)
            model.cuda().eval()
            metrics = details
            macro = sum((metrics[split]["char_f2"] for split in splits)) / 3
            elig = eligible(metrics, baseline, baseline_macro)
            entry = {
                "step": stepno,
                "eligible": bool(elig),
                "macro_source_char_f2": float(macro),
                "metrics": details,
                "optimizer_updates_successful": successful,
                "optimizer_updates_skipped": skipped,
            }
            history.append(entry)
            torch.save(
                {
                    key: value.detach().cpu()
                    for key, value in model.head.state_dict().items()
                },
                out / f"head-step-{stepno}.pt",
            )
            if elig and macro > best:
                best = macro
                best_step = stepno
                best_improved = True
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.head.state_dict().items()
                }
            report["history"] = history
            report["best_step_macro_source_char_f2"] = float(best)
            report["best_improved_over_original_macro"] = bool(best_improved)
            (root / "report.json").write_text(json.dumps(report, indent=2))
    if not best_improved:
        fallback = restore(out, source, deep, PROTOCOL)
        report.update(
            {
                "best_improved_over_original_macro": False,
                "ready": False,
                "selected_model": "frozen_deep_baseline_fallback",
                "fallback_artifact": fallback,
                "selected_head_bias_delta": None,
                "calibration": [],
                "elapsed_seconds": float(time.time() - start),
                "provisional": True,
            }
        )
        (root / "report.json").write_text(json.dumps(report, indent=2))
        print(
            {
                "ready_for_review": False,
                "fallback": "frozen Deep baseline",
                "reason": "no eligible INT8 checkpoint exceeded original source macro",
                "report": str(root / "report.json"),
            },
            flush=True,
        )
        return
    model.head.load_state_dict(best_state)
    raw = out / "raw"
    shutil.rmtree(raw, ignore_errors=True)
    pack(model, tokenizer, raw, torch)
    (raw / "float.onnx").unlink(missing_ok=True)
    init_audit = audit(onnx, source / "base.onnx", raw / "model.onnx")
    init_preserved = (
        init_audit["all_source_tensors_preserved"]
        and init_audit["classifier_bias_tensor_preserved"]
        and (not any(init_audit["duplicate_normalized_names"].values()))
    )
    report["base_initializer_audit"] = init_audit
    shutil.rmtree(raw, ignore_errors=True)
    base_metrics = baseline
    candidates = []
    chosen = None
    old_bias = model.head.bias.detach().clone()
    for delta in PROTOCOL["calibration_head_bias_delta"]:
        model.head.bias.data.copy_(old_bias + float(delta))
        path = out / f"candidate-{delta:g}"
        shutil.rmtree(path, ignore_errors=True)
        pack(model, tokenizer, path, torch)
        (path / "float.onnx").unlink(missing_ok=True)
        sess = ort.InferenceSession(
            str(path / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        results = {
            split: predict(data[split], sess, ort_collate, numpy) for split in splits
        }
        metrics = {
            split: evaluate(
                data[split], results[split][1], results[split][2], base_tags[split]
            )
            for split in splits
        }
        for split in splits:
            metrics[split]["date_off"] = dateoff(
                data[split], results[split][1], results[split][2]
            )
        macro = float(sum((metrics[split]["char_f2"] for split in splits)) / 3)
        elig = eligible(metrics, base_metrics, baseline_macro)
        item = {
            "head_bias_delta": float(delta),
            "macro_source_char_f2": macro,
            "metrics": metrics,
            "eligible": bool(elig),
            "model_sha256": hashlib.sha256(
                (path / "model.onnx").read_bytes()
            ).hexdigest(),
            "model_bytes": (path / "model.onnx").stat().st_size,
        }
        candidates.append(item)
        if elig and (
            chosen is None
            or sum((metrics[split]["char_f2"] for split in splits))
            > sum((chosen["metrics"][split]["char_f2"] for split in splits))
        ):
            chosen = item
        shutil.rmtree(path, ignore_errors=True)
    if chosen is None:
        fallback = restore(out, source, deep, PROTOCOL)
        report.update(
            {
                "history": history,
                "calibration": candidates,
                "best_improved_over_original_macro": bool(best_improved),
                "ready": False,
                "selected_model": "frozen_deep_baseline_fallback",
                "fallback_artifact": fallback,
                "selected_head_bias_delta": None,
                "elapsed_seconds": float(time.time() - start),
                "provisional": True,
            }
        )
        (root / "report.json").write_text(json.dumps(report, indent=2))
        print(
            {
                "ready_for_review": False,
                "fallback": "frozen Deep baseline",
                "reason": "no calibrated residual passed source/gold/date-off gates",
                "report": str(root / "report.json"),
            },
            flush=True,
        )
        return
    model.head.bias.data.copy_(old_bias + chosen["head_bias_delta"])
    torch.save(
        {key: value.detach().cpu() for key, value in model.head.state_dict().items()},
        root / "head.pt",
    )
    report["selected_step"] = best_step
    final = out / "model"
    shutil.rmtree(final, ignore_errors=True)
    pack(model, tokenizer, final, torch)
    (final / "float.onnx").unlink(missing_ok=True)
    final_session = ort.InferenceSession(
        str(final / "model.onnx"), options, providers=["CPUExecutionProvider"]
    )
    headless = out / "headless-probe"
    shutil.rmtree(headless, ignore_errors=True)
    pack(model.base, tokenizer, headless, torch)
    (headless / "float.onnx").unlink(missing_ok=True)
    headless_session = ort.InferenceSession(
        str(headless / "model.onnx"), options, providers=["CPUExecutionProvider"]
    )
    asset_hashes = {
        name: hashlib.sha256((final / name).read_bytes()).hexdigest()
        for name in ("tokenizer.json", "labels.json", "tokens.json")
    }
    asset_ok = (
        asset_hashes["tokenizer.json"] == PROTOCOL["selected_tokenizer_sha256"]
        and asset_hashes["labels.json"] == PROTOCOL["labels_sha256"]
        and (asset_hashes["tokens.json"] == PROTOCOL["tokens_sha256"])
    )
    smoke = []
    for row in data["nv.valid"][:3]:
        entry = windows(row)[0][-1]
        batch = ort_collate(
            [{key: value for key, value in entry.items() if key != "labels"}]
        )
        inputs = {key: value.numpy() for key, value in batch.items()}
        smoke.append((inputs, headless_session.run(None, inputs)[0]))
    smoke_result = smokecheck(final_session, smoke, numpy)
    shutil.rmtree(headless, ignore_errors=True)
    final_bytes = (final / "model.onnx").stat().st_size
    final_sha = hashlib.sha256((final / "model.onnx").read_bytes()).hexdigest()
    report.update(
        {
            "history": history,
            "calibration": candidates,
            "selected_head_bias_delta": chosen["head_bias_delta"],
            "selected_metrics": chosen["metrics"],
            "selected_eligible": chosen["eligible"],
            "best_improved_over_original_macro": bool(best_improved),
            "final_model_sha256": final_sha,
            "final_model_bytes": int(final_bytes),
            "output_asset_sha256": asset_hashes,
            "output_assets_match_pins": bool(asset_ok),
            "base_initializer_audit": init_audit,
            "parameters_trainable": 258,
            "optimizer_updates_successful": int(successful),
            "optimizer_updates_skipped": int(skipped),
            "initial_argmax_flip_count": int(changes),
            "export_smoke": smoke_result,
            "ready": bool(
                best_improved
                and chosen["eligible"]
                and (final_bytes <= 10000000)
                and init_preserved
                and (changes == 0)
                and asset_ok
                and smoke_result["passed"]
            ),
            "elapsed_seconds": float(time.time() - start),
            "provisional": True,
        }
    )
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(
        {
            "ready_for_review": report["ready"],
            "report": str(root / "report.json"),
            "training_launched": True,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
