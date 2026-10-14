"""Check each narrated passage in the exported video against its source audio."""

import argparse
import json
import subprocess
import wave
from pathlib import Path

import numpy
from scipy.signal import correlate

from narrate import FOLDER, RATE, SCRIPT, read


def check(movie):
    """Match each spoken passage against its source waveform."""
    decoded = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(movie), "-map", "0:a:0", "-ac", "1",
        "-ar", str(RATE), "-f", "f32le", "-c:a", "pcm_f32le", "-",
    ])
    audio = numpy.frombuffer(decoded, dtype="<f4")
    assert len(audio) and numpy.isfinite(audio).all()
    timeline = json.loads((FOLDER / "build/timing.json").read_text())
    assert [row["key"] for row in timeline] == list(SCRIPT)
    results = []
    margin = round(0.03 * RATE)
    for row in timeline:
        path, duration, _ = read(row["key"])
        with wave.open(str(path)) as wav:
            source = numpy.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(numpy.float32) / 32768
        start = round(row["start"] * RATE)
        assert start >= margin and start + len(source) + margin <= len(audio)
        actual = audio[start - margin:start + len(source) + margin]
        rms = float(numpy.sqrt(numpy.mean(actual ** 2)))
        # Allow codec delay while matching each spoken waveform.
        matches = correlate(actual, source, mode="valid", method="fft")
        energy = numpy.r_[0, numpy.cumsum(actual.astype(float) ** 2)]
        energy = energy[len(source):] - energy[:-len(source)]
        similarity = float(numpy.max(matches / numpy.sqrt(numpy.maximum(energy * numpy.sum(source ** 2), 1e-15))))
        assert rms > 0.025 and similarity > 0.9, (
            f"Missing or damaged narration: {row['key']!r}; RMS={rms:.4f}, similarity={similarity:.3f}"
        )
        assert abs(row["end"] - row["start"] - duration) < 0.001
        results.append({"key": row["key"], "rms": round(rms, 4), "similarity": round(similarity, 4)})
    report = FOLDER / "build/audio-check.json"
    report.write_text(json.dumps(results, indent=2) + "\n")
    print(f"PASS: all {len(results)} spoken clips are audible and match their sources in {movie.name}.")
    print(f"Lowest speech similarity: {min(row['similarity'] for row in results):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("movie", type=Path, nargs="?", default=FOLDER.parent / "results/Veyl.webm")
    check(parser.parse_args().movie)
