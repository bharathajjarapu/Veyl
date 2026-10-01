"""Generate cached, sentence-timed narration: uv run python narrate.py."""

import argparse
import ast
import hashlib
import json
import re
import wave
from pathlib import Path

import numpy

FOLDER = Path(__file__).resolve().parent
SCRIPT = json.loads((FOLDER / "narration.json").read_text())
REVISION = "cca42f3882e9d36b865883298d8e3c6a6e42b9c0"
MODEL_SHA256 = "f3daed94c8fe1f7a90f14cd7fc8ead55c99a8e57dd0aac669e376882483222c7"
VOICE = "af_heart"
SPEED = 0.96
RATE = 24000


def files(key):
    """Invalidate audio whenever the script, generator or voice changes."""
    signature = json.dumps([SCRIPT[key], REVISION, VOICE, SPEED]) + Path(__file__).read_text()
    name = hashlib.sha256(signature.encode()).hexdigest()[:20]
    path = FOLDER / "audio" / name
    return path.with_suffix(".wav"), path.with_suffix(".json")


def read(key):
    """Load a cached clip and validate its audio and sentence timings."""
    path, metadata = files(key)
    if not path.exists() or not metadata.exists():
        raise FileNotFoundError("Generate narration first: uv run python narrate.py")
    cues = json.loads(metadata.read_text())
    with wave.open(str(path)) as audio:
        assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, RATE)
        duration = audio.getnframes() / RATE
        pcm = numpy.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    assert 0.5 < duration < 60 and numpy.max(numpy.abs(pcm.astype(float))) < 32767
    assert numpy.sqrt(numpy.mean((pcm.astype(float) / 32768) ** 2)) > 0.01
    assert cues and cues[0]["start"] == 0
    assert all(0 <= cue["start"] < cue["end"] <= duration + 1 / RATE for cue in cues)
    assert all(left["end"] <= right["start"] for left, right in zip(cues, cues[1:]))
    assert " ".join(cue["text"] for cue in cues) == SCRIPT[key]
    return path, duration, cues


def check():
    """Check that every scene has a valid narration clip."""
    tree = ast.parse((FOLDER / "film.py").read_text())
    captions = [node.args[0].value for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "stream"]
    assert len(captions) == len(set(captions))
    assert set(captions) | {"intro", "outro"} == set(SCRIPT)
    for key in SCRIPT:
        read(key)
    print(f"Checked {len(SCRIPT)} clips: script coverage, PCM audio, levels and sentence timings.")


def generate():
    """Generate missing narration clips from the pinned voice model."""
    from huggingface_hub import hf_hub_download
    from titl import Voice, write_wav

    model = Path(hf_hub_download("MithrilMan/teacher-in-the-loop",
                               f"{VOICE}/model_int8.onnx", revision=REVISION))
    assert hashlib.sha256(model.read_bytes()).hexdigest() == MODEL_SHA256
    voice = Voice(model, "en", threads=1)
    (FOLDER / "audio").mkdir(exist_ok=True)
    for key, sentence in SCRIPT.items():
        path, metadata = files(key)
        if path.exists() and metadata.exists():
            read(key)
            continue
        parts, cues, samples = [], [], 0
        for phrase in re.split(r"(?<=[.!?])\s+", sentence):
            audio = voice(phrase.replace("Veyl", "Vail"), speed=SPEED)
            assert len(audio) and numpy.isfinite(audio).all()
            active = numpy.flatnonzero(numpy.abs(audio) > 0.006)
            assert len(active), phrase
            audio = audio[max(0, active[0] - 1440):min(len(audio), active[-1] + 2880)]
            fade = min(120, len(audio) // 2)
            audio[:fade] *= numpy.linspace(0, 1, fade)
            audio[-fade:] *= numpy.linspace(1, 0, fade)
            cues.append({"text": phrase, "start": samples / RATE,
                         "end": (samples + len(audio)) / RATE})
            parts.extend([audio, numpy.zeros(4320, dtype=numpy.float32)])
            samples += len(audio) + 4320
        audio = numpy.concatenate(parts[:-1])
        rms = float(numpy.sqrt(numpy.mean(audio ** 2)))
        audio *= min(0.12 / rms, 0.92 / float(numpy.max(numpy.abs(audio))))
        write_wav(path, audio)
        metadata.write_text(json.dumps(cues, indent=2, ensure_ascii=False) + "\n")
        print(f"{len(audio) / RATE:5.1f}s  {key}", flush=True)
    check()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check existing audio without loading TTS")
    if parser.parse_args().check:
        check()
    else:
        generate()
