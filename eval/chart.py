"""Plot pinned PIIMB results; no inference, model changes or network requests."""

import csv
import hashlib
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plot
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/chart"
REPORTS = ROOT / "local/reports"
CHART = REPORTS / "chart"
RAW = ROOT / "local/comparisons"
source = json.loads((CHART / "official.json").read_text())
weights = {
    row["name"]: row
    for row in json.loads((CHART / "weights.json").read_text())
    if "bytes" in row
}
rows = source["models"]
veyl = json.loads((REPORTS / "piimb.json").read_text())
assert len(rows) == 35 and len({row["name"] for row in rows}) == 35
assert set(source["tasks"]) == set(veyl["tasks"])
for row in rows:
    meta = RAW / row["folder"] / "model_meta.json"
    assert hashlib.sha256(meta.read_bytes()).hexdigest() == row["meta_sha256"]
    meta = json.loads(meta.read_text())
    assert (row["name"], row["params"], row["dtype"], row["model_revision"]) == (
        meta["name"],
        meta["n_parameters"],
        meta["dtype"],
        meta.get("revision"),
    )
    if row["name"] in weights:
        assert weights[row["name"]]["revision"] == row["model_revision"]
        assert (
            isinstance(weights[row["name"]]["bytes"], int)
            and weights[row["name"]]["bytes"] > 0
        )
    assert len(row["tasks"]) == 4
    assert (
        abs(row["mean_f2"] - sum((task["f2"] for task in row["tasks"].values())) / 4)
        < 1e-12
    )
    for task in source["tasks"]:
        path = RAW / row["folder"] / (task + ".json")
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == row["details"][task]["sha256"]
        )
        scores = json.loads(path.read_text())["scores"]["test"]
        assert len(scores) == 1 and scores[0]["hf_subset"] == "sentences"
        raw = scores[0]
        assert row["tasks"][task] == {field: raw[field] for field in row["tasks"][task]}
assert (
    abs(
        veyl["unweighted_task_f2_mean"]
        - sum((task["f2"] for task in veyl["tasks"].values())) / 4
    )
    < 1e-12
)
assert veyl["model_only"] and veyl["micro"]["rows"] == 121409
assert veyl["prediction_meta"]["parameters"] == 5170107
assert (ROOT / "weights/model.onnx").stat().st_size == 5426280
assert (
    hashlib.sha256((ROOT / "weights/model.onnx").read_bytes()).hexdigest()
    == veyl["prediction_meta"]["model_sha256"]
)
ours = {
    "name": "Veyl (local INT8)",
    "params": 5170107,
    "bytes": 5426280,
    "mean_f2": veyl["unweighted_task_f2_mean"],
    **{
        f"mean_{field}": sum((task[field] for task in veyl["tasks"].values())) / 4
        for field in ["precision", "recall", "fpr"]
    },
}
models = {row["name"]: row for row in rows}
labels = {
    "OpenMed/privacy-filter-multilingual-v2": "OpenMed multilingual v2",
    "knowledgator/gliner-stream-pii-v1.0": "GLiNER-stream",
    "OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1": "OpenMed Large",
    "antoniogr7/pii-identifier-v0.1": "PII Identifier",
    "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1": "OpenMed Small",
    "kalyan-ks/ettin-17m-nemotron-pii": "Ettin 17M",
    "kalyan-ks/ettin-32m-nemotron-pii": "Ettin 32M",
    "gravitee-io/bert-small-pii-detection": "Gravitee BERT-small",
    "nationaldesignstudio/rampart": "Rampart Q4",
}
blue, orange, grey = ("#2261a0", "#bf5b14", "#8492a0")
plot.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#29384a",
        "text.color": "#1d2c3d",
        "xtick.color": "#42556b",
        "ytick.color": "#42556b",
        "svg.fonttype": "none",
    }
)


def save(figure, name):
    """Save both chart formats and close the figure."""
    for suffix in ["png", "svg"]:
        figure.savefig(
            OUT / f"{name}.{suffix}", dpi=170, bbox_inches="tight", facecolor="white"
        )
    plot.close(figure)


figure, (left, right) = plot.subplots(
    1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.08, 1]}
)
figure.suptitle(
    "Veyl vs published PIIMB English results",
    fontsize=20,
    fontweight="bold",
    x=0.03,
    ha="left",
)
selected = [models[name] for name in labels] + [ours]
selected.sort(key=lambda row: row["mean_f2"])
for index, row in enumerate(selected):
    local = row is ours
    left.barh(index, row["mean_f2"] * 100, height=0.58, color=orange if local else blue)
    left.text(
        row["mean_f2"] * 100 + 0.6,
        index,
        f"{row['mean_f2'] * 100:.2f}",
        va="center",
        fontsize=12,
        fontweight="bold" if local else "normal",
    )
left.set_yticks(
    range(len(selected)), [labels.get(row["name"], row["name"]) for row in selected]
)
left.set_xlim(0, 102)
left.set_xlabel("Task-mean character masking F2 (%)")
left.set_title("Selected published systems + Veyl local result", loc="left", pad=14)
left.grid(axis="x", alpha=0.16)
left.set_axisbelow(True)
offsets = {
    "Rampart Q4": (10, -8),
    "Ettin 17M": (10, -18),
    "Ettin 32M": (-20, 14),
    "Gravitee BERT-small": (12, -14),
    "PII Identifier": (-50, 12),
    "OpenMed Small": (-54, -24),
    "OpenMed Large": (-55, -24),
    "OpenMed multilingual v2": (-168, 10),
}
for row in selected:
    local = row is ours
    if not local and row["name"] not in weights:
        continue
    size = ours["bytes"] if local else weights[row["name"]]["bytes"]
    name = "Veyl INT8" if local else labels[row["name"]]
    right.scatter(
        size / 1000000.0,
        row["mean_f2"] * 100,
        s=100 if local else 58,
        color=orange if local else blue,
        marker="*" if local else "o",
        zorder=4,
    )
    right.annotate(
        name,
        (size / 1000000.0, row["mean_f2"] * 100),
        xytext=(10, 7) if local else offsets[name],
        textcoords="offset points",
        fontsize=11,
        fontweight="bold" if local else "normal",
        arrowprops={"arrowstyle": "-", "color": grey, "lw": 0.7},
    )
right.set_xscale("log")
right.set_xticks([5, 10, 30, 100, 300, 1000, 3000])
right.xaxis.set_major_formatter(ScalarFormatter())
right.set_xlim(3.5, 4400)
right.set_ylim(56, 97)
right.set_xlabel("Recorded model / checkpoint artifact (MB, log scale)")
right.set_ylabel("Task-mean masking F2 (%)")
right.set_title("Storage footprint vs masking quality", loc="left", pad=14)
right.grid(alpha=0.16)
figure.text(
    0.03,
    -0.01,
    "Blue: published PIIMB pipelines. Orange: our local WASM run, not a leaderboard entry. Four English tasks, PIIMB 0.3.0.\nDisk bytes exclude tokenizers/runtime; formats differ (Veyl INT8, Rampart Q4, others FP32/BF16). Disk size is not RAM or compute.",
    fontsize=11,
)
figure.subplots_adjust(top=0.84, bottom=0.14, wspace=0.45)
save(figure, "quality")
figure, (left, right) = plot.subplots(1, 2, figsize=(16, 7))
figure.suptitle(
    "Precision, recall and measured local inference cost",
    fontsize=20,
    fontweight="bold",
    x=0.03,
    ha="left",
)
left.scatter(
    [row["mean_precision"] * 100 for row in rows],
    [row["mean_recall"] * 100 for row in rows],
    color=grey,
    s=35,
    alpha=0.7,
    label="35 published pipelines",
)
for name, shift in [
    ("OpenMed/privacy-filter-multilingual-v2", (-150, 8)),
    ("gravitee-io/bert-small-pii-detection", (-168, -24)),
    ("nationaldesignstudio/rampart", (-115, -20)),
]:
    row = models[name]
    left.scatter(
        row["mean_precision"] * 100, row["mean_recall"] * 100, color=blue, s=62
    )
    left.annotate(
        labels[name],
        (row["mean_precision"] * 100, row["mean_recall"] * 100),
        xytext=shift,
        textcoords="offset points",
        fontsize=11,
        arrowprops={"arrowstyle": "-", "color": grey, "lw": 0.7},
    )
left.scatter(
    ours["mean_precision"] * 100,
    ours["mean_recall"] * 100,
    color=orange,
    s=150,
    marker="*",
    label="Veyl local",
)
left.annotate(
    "Veyl: P 78.04 / R 91.50",
    (ours["mean_precision"] * 100, ours["mean_recall"] * 100),
    xytext=(-158, -23),
    textcoords="offset points",
    fontsize=11,
    fontweight="bold",
    arrowprops={"arrowstyle": "-", "color": grey, "lw": 0.7},
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1},
)
left.set_xlim(45, 100)
left.set_ylim(32, 100)
left.set_xlabel("Mean per-task precision (%)")
left.set_ylabel("Mean per-task recall (%)")
left.set_title("F2 alone hides the false-mask tradeoff", loc="left", pad=14)
left.grid(alpha=0.16)
left.legend(loc="lower left", frameon=False, fontsize=10)
local = []
for name in ["veyl", "tall", "wide"]:
    path = REPORTS / (
        "speed.json" if name == "veyl" else f"chart/{name}.speed.json"
    )
    speed = json.loads(path.read_text())
    score = (
        veyl
        if name == "veyl"
        else json.loads((CHART / f"{name}.score.json").read_text())
    )
    fixture = next(
        (
            measure
            for measure in speed["measurements"]
            if measure["name"] == "full_window_510"
        )
    )
    assert (
        fixture["token_pieces"] == 510
        and fixture["input_chars"] == 1019
        and (speed["threads"] == 1)
    )
    row = {
        "name": name,
        "latency_ms": fixture["p50_ms"],
        "f2": score["unweighted_task_f2_mean"],
        "speed_source": str(path.relative_to(ROOT)),
        "timestamp": speed["completed_at"],
    }
    local.append(row)
    right.scatter(
        row["latency_ms"],
        row["f2"] * 100,
        s=140 if name == "veyl" else 70,
        marker="*" if name == "veyl" else "o",
        color=orange if name == "veyl" else grey,
    )
    right.annotate(
        f"{name.title()}\n{row['latency_ms']:.1f} ms / {row['f2'] * 100:.2f} F2",
        (row["latency_ms"], row["f2"] * 100),
        xytext=(12, 12) if name != "wide" else (-125, -35),
        textcoords="offset points",
        fontsize=12,
    )
right.set_xlim(60, 230)
right.set_ylim(87.8, 88.9)
right.set_xlabel("Full-window inference p50 (ms)")
right.set_ylabel("Task-mean masking F2 (%), zoomed axis")
right.set_title(
    "Our variants: same PC / runtime / 510-piece fixture", loc="left", pad=14
)
right.grid(alpha=0.16)
figure.text(
    0.03,
    -0.01,
    "Left: arithmetic means over four tasks; not pooled micro P/R. Right: Chrome 152, one WASM thread, separate recorded runs.\nExternal latency / RAM / energy are not reported by PIIMB. No GPU was required for these browser runs. Training cost is not plotted.",
    fontsize=11,
)
figure.subplots_adjust(top=0.84, bottom=0.14, wspace=0.3)
save(figure, "cost")
with (OUT / "table.csv").open("w") as stream:
    writer = csv.writer(stream)
    writer.writerow(
        [
            "model",
            "status",
            "parameters",
            "artifact_bytes",
            "mean_f2",
            "mean_precision",
            "mean_recall",
            "mean_fpr",
        ]
    )
    for row in [ours, *rows]:
        writer.writerow(
            [
                row["name"],
                "local" if row is ours else "published",
                row["params"],
                row["bytes"]
                if row is ours
                else weights.get(row["name"], {}).get("bytes", ""),
                *[
                    row[field]
                    for field in [
                        "mean_f2",
                        "mean_precision",
                        "mean_recall",
                        "mean_fpr",
                    ]
                ],
            ]
        )
(CHART / "local.json").write_text(
    json.dumps(
        {
            "veyl": ours,
            "latency": local,
            "plot_versions": {"matplotlib": matplotlib.__version__},
        },
        indent=2,
    )
    + "\n"
)
print(
    "Verified raw source hashes and computed task means; wrote quality/cost PNG+SVG and 36-row CSV."
)

audit = {
    "scope": "Verified comparison sources and generated figures. No model inference.",
    "published_models": len(rows),
    "english_task_files": len(rows) * len(source["tasks"]),
    "model_metadata_files": len(rows),
    "mean_recalculation_max_error": 0,
    "source_revision": source["source_revision"],
    "model_sha256": veyl["prediction_meta"]["model_sha256"],
    "files": {},
}
for path in [
    Path(__file__),
    ROOT / "docs/eval.md",
    *[CHART / name for name in ["official.json", "weights.json", "local.json"]],
    *[
        OUT / name
        for name in [
            "quality.png",
            "quality.svg",
            "cost.png",
            "cost.svg",
            "table.csv",
        ]
    ],
]:
    audit["files"][str(path.relative_to(ROOT))] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
(CHART / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
