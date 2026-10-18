# Veyl Video

Narrated Manim animation of the Veyl pipeline. Reads FP32 weights with NumPy; no PyTorch or training.

| File | Purpose |
|---|---|
| `film.py` | Animation |
| `narration.json` | Narration script |
| `narrate.py` | Voice clip generation |
| `soundtrack.py` | Score and final mix |
| `../results/Veyl.webm` | Output |

## Setup

Requires `../weights/` and the Git-ignored `../local/reports/`.

```sh
sudo apt install build-essential pkg-config python3-dev libcairo2-dev libpango1.0-dev ffmpeg
uv sync --locked --python 3.12
```

## Render

```sh
uv run --locked python narrate.py
uv run --locked python film.py
```

Output: `../results/Veyl.webm` (1920×1080, 60 fps, VP9, stereo Opus) and `../results/Veyl.srt`.

Preview (480p, narration only):

```sh
uv run manim -ql --media_dir build film.py Veyl
```

## Narration

Voice: [Teacher in the Loop](https://github.com/MithrilMan/teacher-in-the-loop) `af_heart` (Apache-2.0), distilled from [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). Model revision and SHA-256 are pinned in `narrate.py`; weights come from [Hugging Face](https://huggingface.co/MithrilMan/teacher-in-the-loop).

After editing `narration.json`, rerun both commands. Only changed clips regenerate.

```sh
uv run --locked python narrate.py --check
```

## Music

`soundtrack.py` generates the score locally (75 BPM, no samples). It ducks under narration and targets -16 LUFS, -1.5 dB true peak.

## Verify

```sh
uv run --locked python check.py
```

Decodes the WebM and compares each narration passage with its source. Output: `build/audio-check.json`.

## Notes

- Vectors and attention use the FP32 checkpoint on one fixed example; training scores are illustrative.
- Masking F2 comes from `../local/reports/fp32.json` and `../local/reports/piimb.json`. Different runtimes and tokenizers mean the gap does not isolate quantization loss.
- Asset credits and font licenses: `assets/`.
