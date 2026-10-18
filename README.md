# VEYL

Vigilant Entity Yield Limiter

VEYL detects English personal information and returns character spans for redaction. A BERT student learns from a trained teacher. A separate AGE head changes age predictions only where the student predicts O.

| Model | Value |
| --- | --- |
| Encoder | 6 BERT layers, width 128, 2 attention heads |
| Tokenizer | 30,522 uncased WordPiece tokens |
| Output | 57 BIO labels across 28 PII categories |
| Parameters | 5,170,107, including the 258-parameter AGE head |
| INT8 ONNX | 5,426,280 bytes |
| ONNX, tokenizer and labels | 6,138,595 bytes |

https://github.com/user-attachments/assets/c2277414-be1c-4550-831b-134bef527be0

## Setup

Run from this directory with Python 3.12 and uv. Input-dependent commands require the Git-ignored `local/` archive, prepared datasets and FP32 checkpoints. A Git clone contains the deployment bundle but excludes those inputs.

```sh
uv sync --locked --python 3.12
uv run --locked python -m train.check
uv run --locked python data/check.py
uv run --locked python eval/score.py --check
```

The default environment requires NumPy. [Training](docs/TRAIN.md) covers PyTorch setup, FP32 inference and ONNX reconstruction. The [notebook](notebooks/model.ipynb) runs the CPU checks.

## Evaluation

| Dataset | Masking F2 |
| --- | ---: |
| PIIMB English, four-task mean | 88.03% |
| Real-120 | 93.75% |
| TAB, pooled annotator rows | 67.84% |
| RedactionBench, mandatory policy | 68.56% |

These local scores use INT8 ONNX Runtime Web with one WASM thread. PIIMB covers 121,409 English rows. Its mean task precision and recall are 78.04% and 91.50%. Dataset policies differ. See [evaluation](docs/EVAL.md) and the [comparison CSV](results/chart/table.csv).

The model misses PII and can redact ordinary words or organization names. Mixed-script boundaries remain unreliable. Other runtimes require separate evaluation. The cleaned training recipes passed CPU and gradient checks, but have not completed another full training run.

## Files and licenses

`src/` provides inference and export; `train/` contains training recipes; `eval/` contains benchmark tools. `weights/` holds model assets. `results/` contains charts, the CSV and video. Detailed reports, predictions and comparison sources stay Git-ignored in `local/`. Browser code lives in the separate Web repository.

| Data source | Recorded license |
| --- | --- |
| [NVIDIA Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) | CC-BY-4.0 |
| [AI4Privacy OpenPII](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1m) | CC-BY-4.0 |
| [PIIMB](https://huggingface.co/datasets/piimb/pii-masking-benchmark) | CC-BY-NC-4.0 |
| [TAB](https://github.com/NorskRegnesentral/text-anonymization-benchmark) | MIT |
| [RedactionBench](https://huggingface.co/datasets/RedactionBench/RedactionBench) | CC-BY-4.0 |

Manifests in `data/` record revisions, hashes and attribution. No project code or model license has been selected.

---

Made with ❤️ and 💻 by [Bharath Ajjarapu](https://github.com/bharathajjarapu).
