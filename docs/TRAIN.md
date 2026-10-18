# Training

Run from the Veyl root. Python 3.11 and 3.12 are supported. Training requires CUDA; inference and checks run on CPU.

## Local checks

Requires the Git-ignored `local/` archive, prepared datasets and FP32 checkpoints. `local/reports/files.json` pins their hashes.

```sh
uv sync --locked --python 3.12
uv run --locked python -m train.check
uv run --locked python data/check.py
```

The default environment installs NumPy. The `train` extra adds Transformers 4.x; `onnx` adds ONNX and ONNX Runtime. Install PyTorch separately. `uv.lock` does not reproduce the original Kaggle CUDA environment.

## Kaggle setup

Copy Veyl and its retained inputs to `/kaggle/working/Veyl`, then run:

```sh
python -m pip install uv
uv pip install --system -r pyproject.toml --extra train --extra onnx
python -m train.check --torch
python -m eval.model
python -m src.model 'Contact John Smith at john@example.com.'
python -m train.deep
python -m train.age
```

Training rejects existing `runs/deep/` and `runs/age/` directories. AGE starts from the retained Deep checkpoint. Neither command replaces shipped weights.

## Optimization

- Teacher: BERT-base, 3 epochs.
- Deep: 6 layers, width 128, 2 heads, distilled for 4 epochs. Pretrained revisions are pinned in the [job manifest](../data/broad/job.json).
- Distillation loss: `(CE + T² × KL) / 2`, with `T = 2`. Supervision uses the first subword; ignored labels are `-100`.
- AGE: freezes Deep, subtracts 1.0 from its O logit, and adds a 258-parameter linear head with a nonnegative Softplus residual. Loss combines weighted positive cross-entropy and marginal O loss for incomplete annotations. Deployment applies the residual only where calibrated Deep predicts O.
- AGE run: seed 7, batch 32, sequence length 512, 1,200 attempted steps. Validation selects step 900 with zero extra head bias. Scheduler steps follow successful optimizer updates.

## Data

| Prepared input | Training rows |
| --- | ---: |
| Broad | 35,485; 35,207 pass boundary checks |
| OpenPII | 4,389 |
| Luna-authored | 128 |
| Benign | 128 |

Tune also includes 117 Haiku validation rows and shares NVIDIA files with Broad. The [Broad](../data/broad/data.json) and [Tune](../data/tune/data.json) manifests record sampling and hashes. Recipes require these exact snapshots; collection scripts are excluded.

Incomplete annotations mask unlabelled classes. `data/check.py` verifies hashes and span bounds, not label correctness. Synthetic labels and overlapping source families limit independence. Checkpoint selection excludes PIIMB, Real-120, TAB and RedactionBench, but developers inspected their results.

## Inference and export

FP32 inference returns Unicode code point spans, using word-aligned 510-piece windows, CLS/SEP tokens, 64-piece overlap and BIO decoding.

```sh
uv run --locked --extra onnx python src/build.py --check
uv run --locked --extra onnx python src/build.py runs/export
```

The builder reconstructs the pinned INT8 graph from `weights/source.onnx` and rejects different destination files. Fresh PyTorch export and quantization (`train/common.py`) need separate evaluation.

The teacher checkpoint is absent. CPU and FP32 gradient checks passed; a new full training run is unverified and may produce different weights.
