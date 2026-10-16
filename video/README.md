# Veyl video

The narrated Manim animation explains the datasets, teacher training, distillation, AGE gate, BERT inference and INT8 export. It uses NumPy to read the retained FP32 weights. It does not install PyTorch or train the model.

The animation is in [film.py](film.py), the editable script is in [narration.json](narration.json), and [narrate.py](narrate.py) generates the voice clips. [soundtrack.py](soundtrack.py) composes the background score and mixes the final audio. The output is [results/Veyl.webm](../results/Veyl.webm).

## Run

Keep this folder inside Veyl. The generator reads `../weights/` and the Git-ignored `../local/reports/`, including the pinned FP32 and INT8 benchmark reports. Supply those local inputs separately after a Git clone.

On Ubuntu or Debian, install the native rendering dependencies once.

```sh
sudo apt install build-essential pkg-config python3-dev libcairo2-dev libpango1.0-dev ffmpeg
```

Open a terminal in this `video/` folder, then run:

```sh
uv sync --locked --python 3.12
uv run --locked python narrate.py
uv run --locked python film.py
```

The commands write `../results/Veyl.webm` at 1920 by 1080 and 60 fps, with VP9 video, stereo Opus audio and an optional English WebVTT subtitle track. The sidecar subtitles are in `../results/Veyl.srt`. The encoder checks each spoken clip before replacing the previous video. Manim keeps temporary renders and scene timings in `build/`. Rendering and speech synthesis use the CPU. Initial setup downloads dependencies and the voice model.

Git tracks `results/Veyl.webm`. Sidecar subtitles, local reports, `audio/` clips, render caches and the local environment remain ignored.

For a quick 480p preview:

```sh
uv run manim -ql --media_dir build film.py Veyl
```

Generate narration first. The preview includes speech and stays under `build/videos/film/480p15/`; the final command adds music, embeds subtitles and normalizes loudness. The bundled fonts and `fonts.conf` keep system font substitutions out of the render. Asset credits and font licenses are in `assets/`.

## Voice and synchronization

The voice is the free, Apache-2.0 [Teacher in the Loop](https://github.com/MithrilMan/teacher-in-the-loop) English `af_heart` student, a follow-up to [Paradee](https://github.com/sahilmahendrakar/paradee), distilled from [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). It has about 8 million parameters; “9 MB” describes its INT8 ONNX file, not its parameter count or the full Python environment. The model revision and SHA-256 are pinned in `narrate.py`, and Python dependencies, including the English pronunciation frontend, are locked in `uv.lock`. Model weights download from [Hugging Face](https://huggingface.co/MithrilMan/teacher-in-the-loop).

Each diagram starts its own voice clip and stays visible until the clip finishes. Sentence timings come from the actual generated audio, and drive the subtitles. Animation transitions cannot interrupt a sentence or overlap the next clip. Sound is inserted directly into the file writer because Manim's scene-level sound API skips it after cached animations. Speech has short sentence pauses, consistent clip levels, and final loudness normalization. Veyl is pronounced “Vail.”

The original background score is composed locally in `soundtrack.py`: gentle electric keys and sustained pads at 75 BPM, with no drums, vocals, downloaded recordings or paid service. It fades in and out, and a sidechain compressor lowers it during narration. The final mix targets -16 LUFS with a -1.5 dB true-peak ceiling, in stereo Opus at 48 kHz.

After editing `narration.json`, rerun both generation and rendering commands. Changed clips regenerate automatically; unchanged clips stay cached. To check the existing clips without loading the voice model:

```sh
uv run --locked python narrate.py --check
```

This checks coverage of every scene, valid audio, signal levels and sentence timings. Rendering also checks for overlapping clips and verifies the retained model and benchmark hashes. `build/timing.json` records the actual narration intervals for review. The short captions above the diagrams summarize the spoken explanation; full subtitles are optional in your player.

To check the actual exported audio:

```sh
uv run --locked python check.py
```

This decodes the WebM and compares every spoken passage with its source waveform, allowing for codec delay. Quiet or missing narration fails even if background music is present. This check also runs before the encoder replaces the previous video; results are saved in `build/audio-check.json`.

## What the diagrams show

Vectors and attention use the retained FP32 checkpoint for the fixed example. Training scores are illustrations. The shrinking quantization grid illustrates lower storage with the same number of weights. The video reads measured masking F2 from `../local/reports/fp32.json` and `../local/reports/piimb.json`. These are local English PIIMB results with different runtime and tokenizer paths. Their difference does not isolate quantization loss. Some ONNX tensors remain FP32.
