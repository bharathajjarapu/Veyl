"""Score saved spans with local PIIMB 0.3.0 character-mask metrics."""

import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eval"
PRED = ROOT / "local" / "predictions"
OUT = ROOT / "runs" / "score"
TASKS = ("ai4privacy-en", "nemotron-pii", "gretel", "privy")
TASK_ROWS = {
    "ai4privacy-en": 18_538,
    "nemotron-pii": 77_907,
    "gretel": 19_601,
    "privy": 5_363,
}
EXPECTED_ROWS = {"real": 120, "tab": 555, "redaction": 200, "piimb": 121_409}
SOURCE_ROWS = 150_022


def sha(path):
    """Hash a file without loading it all into memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rows(path):
    """Read nonblank JSONL records and report malformed line numbers."""
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{number}: invalid JSON") from error
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number}: expected an object")
                yield value


def merge(intervals):
    """Return the union of half-open intervals."""
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def size(intervals):
    """Count covered character positions in disjoint intervals."""
    return sum(end - start for start, end in intervals)


def overlap(left, right):
    """Count shared character positions in two sorted interval unions."""
    total = 0
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if end > start:
            total += end - start
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return total


def span_intervals(spans, length):
    """Validate and union half-open character spans."""
    result = []
    for span in spans:
        start, end = span.get("start"), span.get("end")
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= length
        ):
            raise ValueError(f"invalid span for text length {length}: {span}")
        result.append((start, end))
    return merge(result)


def metric(matched, predicted, gold, text):
    """Compute pooled character precision, recall, F scores, and FPR."""
    precision = matched / predicted if predicted else 0.0
    recall = matched / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = (
        5 * precision * recall / (4 * precision + recall)
        if 4 * precision + recall
        else 0.0
    )
    false_positive = predicted - matched
    non_pii = text - gold
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "fpr": false_positive / non_pii if non_pii > 0 else 0.0,
        "support_chars": gold,
    }


class Score:
    """Pool character counts across rows before computing ratios."""

    def __init__(self):
        """Start an empty character-count accumulator."""
        self.matched = self.pred = self.gold = self.text = self.rows = 0

    def add(self, text, predicted, gold):
        """Add one row after merging its predicted and gold spans."""
        pred = span_intervals(predicted, len(text))
        true = span_intervals(gold, len(text))
        self.pred += size(pred)
        self.gold += size(true)
        self.matched += overlap(pred, true)
        self.text += len(text)
        self.rows += 1

    def result(self):
        """Return metrics for all rows added so far."""
        return {"rows": self.rows, **metric(self.matched, self.pred, self.gold, self.text)}


def subtract(intervals, exclusions):
    """Remove the exclusion union from intervals."""
    cuts = merge(exclusions)
    result = []
    for start, end in merge(intervals):
        parts = [(start, end)]
        for cut_start, cut_end in cuts:
            next_parts = []
            for left, right in parts:
                if cut_end <= left or cut_start >= right:
                    next_parts.append((left, right))
                else:
                    if left < cut_start:
                        next_parts.append((left, cut_start))
                    if cut_end < right:
                        next_parts.append((cut_end, right))
            parts = next_parts
            if not parts:
                break
        result.extend(parts)
    return merge(result)


def spans(row, key="spans"):
    """Return a row's span list or raise on a malformed record."""
    value = row.get(key)
    if not isinstance(value, list):
        raise ValueError(f"expected {key} list")
    return value


def check_manifest(name, source_path):
    """Verify the prepared source file and its pinned raw source."""
    manifest = json.loads((DATA / f"{name}.source.json").read_text(encoding="utf-8"))
    source_hash = sha(source_path)
    hash_key = {
        "real": "sha256",
        "tab": "eval_sha256",
        "redaction": "converted_sha256",
        "piimb": "sha",
    }[name]
    if manifest.get(hash_key) != source_hash:
        raise ValueError(f"{name} source hash does not match its manifest")
    if name == "tab":
        raw = DATA / "tab.raw.json"
        if sha(raw) != manifest.get("sha256"):
            raise ValueError("TAB raw source hash does not match its manifest")
        raw_rows = json.loads(raw.read_text(encoding="utf-8"))
        if len(raw_rows) != manifest.get("documents"):
            raise ValueError("TAB raw document count does not match its manifest")
        if EXPECTED_ROWS[name] != manifest.get("annotator_document_rows"):
            raise ValueError("TAB evaluation row count does not match its manifest")
    elif name == "redaction":
        if sha(DATA / "redaction.raw.jsonl") != manifest.get("raw_sha256"):
            raise ValueError(
                "RedactionBench raw source hash does not match its manifest"
            )
        if EXPECTED_ROWS[name] != manifest.get("rows"):
            raise ValueError("RedactionBench row count does not match its manifest")
    elif name == "real" and EXPECTED_ROWS[name] != manifest.get("rows"):
        raise ValueError("Real challenge row count does not match its manifest")
    return source_hash, manifest


def pred_meta(name, source_hash, allowed_counts, prediction_path=None):
    """Validate prediction metadata and return its file, record, and row count."""
    path = Path(prediction_path) if prediction_path else PRED / f"{name}.jsonl"
    if not path.is_absolute():
        path = ROOT / path
    meta_path = PRED / f"{name}.meta.json"
    if prediction_path:
        meta_path = path.with_suffix(".meta.json")
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"missing {name} predictions or metadata under {PRED}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    count = meta.get("rows", meta.get("count"))
    recorded_hash = meta.get("dataset_sha256", meta.get("source_sha256"))
    if count not in allowed_counts:
        raise ValueError(
            f"{name} prediction metadata has unexpected row count: {count}"
        )
    if recorded_hash != source_hash:
        raise ValueError(f"{name} predictions were made from a different source")
    if meta.get("dataset") not in (None, name):
        raise ValueError(f"{name} prediction metadata names a different dataset")
    if (
        meta.get("model_sha256")
        != json.loads((ROOT / "weights" / "sha.json").read_text(encoding="utf-8"))[
            "model.onnx"
        ]
    ):
        raise ValueError(f"{name} predictions do not use the pinned model")
    if name == "piimb":
        if meta.get("source_rows", SOURCE_ROWS) != SOURCE_ROWS:
            raise ValueError(
                "PIIMB prediction metadata has an unexpected source row count"
            )
        selection = meta.get("selection")
        if selection not in (None, "all", "piimb"):
            raise ValueError("unknown PIIMB prediction selection")
        if selection == "piimb" and count != EXPECTED_ROWS[name]:
            raise ValueError("PIIMB-selected inference has an unexpected row count")
        if selection == "all" and count != SOURCE_ROWS:
            raise ValueError("full-source PIIMB inference has an unexpected row count")
    elif meta.get("source_rows", EXPECTED_ROWS[name]) != EXPECTED_ROWS[name]:
        raise ValueError(
            f"{name} prediction metadata has an unexpected source row count"
        )
    if meta.get("output_sha256") and meta["output_sha256"] != sha(path):
        raise ValueError(f"{name} prediction file hash differs from its metadata")
    return path, meta, count


def match_pred(pred, expected_row, expected_id, task=None):
    """Require the prediction's row, ID, and optional task to match its source."""
    if "row" in pred:
        if pred["row"] != expected_row:
            raise ValueError(f"prediction row mismatch at row {expected_row}")
    elif "row_index" in pred:
        if pred["row_index"] != expected_row - 1:
            raise ValueError(f"prediction index mismatch at row {expected_row}")
    else:
        raise ValueError(f"prediction missing row index at row {expected_row}")
    if str(pred.get("id")) != str(expected_id):
        raise ValueError(f"prediction ID mismatch at row {expected_row}")
    if task is not None and pred.get("task") not in (None, task):
        raise ValueError(f"prediction task mismatch at row {expected_row}")


def evaluate(name, prediction_path=None):
    """Score one retained evaluation source against saved predictions."""
    source_path = DATA / f"{name}.jsonl"
    source_hash, manifest = check_manifest(name, source_path)
    possible = [EXPECTED_ROWS[name]]
    if name == "piimb":
        possible.append(SOURCE_ROWS)
    pred_path, meta, pred_count = pred_meta(
        name, source_hash, possible, prediction_path
    )
    predictions = iter(rows(pred_path))
    result = {
        "dataset": name,
        "source_sha256": source_hash,
        "prediction_sha256": sha(pred_path),
        "scorer_sha256": sha(Path(__file__)),
        "prediction_meta": meta,
        "model_only": True,
    }

    if name == "piimb":
        selected = Score()
        tasks = {task: Score() for task in TASKS}
        task_rows = Counter()
        filtered_predictions = pred_count == EXPECTED_ROWS[name]
        prediction_row = source_row = 0
        for source_row, src in enumerate(rows(source_path), 1):
            selected_row = src.get("task_name") in TASKS and src.get("language") == "en"
            if not filtered_predictions:
                pred = next(predictions, None)
                if pred is None:
                    raise ValueError(
                        f"missing PIIMB prediction at source row {source_row}"
                    )
                match_pred(pred, source_row, src.get("uid"), src.get("task_name"))
            if not selected_row:
                continue
            task = src["task_name"]
            if filtered_predictions:
                prediction_row += 1
                pred = next(predictions, None)
                if pred is None:
                    raise ValueError(
                        f"missing PIIMB prediction at evaluated row {prediction_row}"
                    )
                match_pred(pred, prediction_row, src.get("uid"), task)
            prediction = pred
            predicted = spans(prediction)
            gold = spans(src, "entities")
            selected.add(src["text"], predicted, gold)
            tasks[task].add(src["text"], predicted, gold)
            task_rows[task] += 1
        if source_row != SOURCE_ROWS:
            raise ValueError("PIIMB source row count changed")
        if next(predictions, None) is not None:
            raise ValueError("extra PIIMB prediction rows")
        if selected.rows != EXPECTED_ROWS[name] or dict(task_rows) != TASK_ROWS:
            raise ValueError(f"PIIMB English task counts changed: {dict(task_rows)}")
        per_task = {task: tasks[task].result() for task in TASKS}
        result.update(
            {
                "rows": selected.rows,
                "tasks": per_task,
                "micro": selected.result(),
                "unweighted_task_f2_mean": statistics.mean(
                    per_task[task]["f2"] for task in TASKS
                ),
                "unweighted_task_precision_mean": statistics.mean(
                    per_task[task]["precision"] for task in TASKS
                ),
                "unweighted_task_recall_mean": statistics.mean(
                    per_task[task]["recall"] for task in TASKS
                ),
                "task_order": list(TASKS),
                "metric_note": "Local PIIMB 0.3.0 character-mask calculation. It is not an official leaderboard result.",
            }
        )
    else:
        score = Score()
        all_marked = Score() if name == "redaction" else None
        policy = Score() if name == "redaction" else None
        source_kinds = Counter()
        document_ids = set()
        for number, (src, pred) in enumerate(
            zip(rows(source_path), predictions, strict=True), 1
        ):
            expected_id = src.get("id", src.get("uid", str(number)))
            match_pred(pred, number, expected_id)
            text = src["text"]
            predicted = spans(pred)
            gold = spans(src)
            score.add(text, predicted, gold)
            if name == "redaction":
                mandatory = [
                    span
                    for span in gold
                    if span.get("policy", span.get("label")) == "mandatory"
                ]
                contextual = [
                    span
                    for span in gold
                    if span.get("policy", span.get("label")) == "contextual"
                ]
                if not mandatory and not contextual:
                    raise ValueError(
                        f"RedactionBench row {number} has no recognized policy spans"
                    )
                mandatory_intervals = span_intervals(mandatory, len(text))
                contextual_intervals = span_intervals(contextual, len(text))
                optional = subtract(contextual_intervals, mandatory_intervals)
                predicted_intervals = span_intervals(predicted, len(text))
                effective = subtract(predicted_intervals, optional)
                all_marked.add(text, predicted, gold)
                policy.add(
                    text,
                    [{"start": start, "end": end} for start, end in effective],
                    mandatory,
                )
                source_kinds[bool(src["is_synthetic"])] += 1
            elif name == "tab":
                document_ids.add(src["document_id"])
        if next(predictions, None) is not None:
            raise ValueError(f"extra {name} prediction rows")
        if score.rows != EXPECTED_ROWS[name]:
            raise ValueError(f"{name} evaluated row count changed")
        result["rows"] = score.rows
        if name == "redaction":
            if source_kinds != {False: 101, True: 99}:
                raise ValueError(
                    f"RedactionBench source mix changed: {dict(source_kinds)}"
                )
            result["all_marked_standard"] = all_marked.result()
            result["mandatory_policy_aware"] = policy.result()
            result["policy_note"] = (
                "Mandatory spans are gold. Predictions inside contextual-only spans are ignored for precision. This is the policy-aware primary score."
            )
        else:
            result["overall"] = score.result()
            if name == "tab":
                result["unique_documents"] = len(document_ids)
                if result["unique_documents"] != manifest.get("documents"):
                    raise ValueError("TAB unique document count changed")
                result["statistical_note"] = (
                    "Annotator-document rows repeat documents. The pooled result is descriptive and rows are not statistically independent."
                )

    result["rows"] = EXPECTED_ROWS[name] if name != "piimb" else result["rows"]
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / f"{name}.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def self_check():
    """Check interval unions, pooled metrics, exclusions, and fixed row counts."""
    assert size(merge([(0, 3), (2, 5), (5, 6)])) == 6
    score = Score()
    score.add(
        "abcdefgh",
        [{"start": 0, "end": 3}, {"start": 2, "end": 5}],
        [{"start": 1, "end": 4}],
    )
    got = score.result()
    assert got["precision"] == 3 / 5
    assert got["recall"] == 1.0
    assert abs(got["f1"] - 0.75) < 1e-12
    assert abs(got["f2"] - 15 / 17) < 1e-12
    assert abs(got["fpr"] - 2 / 5) < 1e-12
    contextual = merge([(0, 8)])
    mandatory = merge([(3, 5)])
    optional = subtract(contextual, mandatory)
    effective = subtract(merge([(0, 8)]), optional)
    policy = Score()
    policy.add(
        "abcdefgh",
        [{"start": start, "end": end} for start, end in effective],
        [{"start": 3, "end": 5}],
    )
    assert policy.result()["precision"] == policy.result()["recall"] == 1.0
    assert sum(TASK_ROWS.values()) == EXPECTED_ROWS["piimb"]
    assert 101 + 99 == EXPECTED_ROWS["redaction"]
    assert span_intervals([], 0) == []
    print("score self-check passed")


def main(args):
    """Run a dataset score, an alternate prediction file, or the self-check."""
    if args == ["--check"]:
        self_check()
        return
    if len(args) not in (1, 2) or args[0] not in EXPECTED_ROWS:
        raise SystemExit(
            "Usage: python3 eval/score.py real|tab|redaction|piimb [prediction.jsonl] | --check"
        )
    self_check()
    evaluate(args[0], args[1] if len(args) == 2 else None)


if __name__ == "__main__":
    main(sys.argv[1:])
