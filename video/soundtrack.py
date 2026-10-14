"""An original, deterministic ambient score and the final voice-first mix."""

import argparse
import json
import subprocess
import wave
from pathlib import Path

import numpy

from check import check
from narrate import FOLDER, RATE


def score(duration):
    """Soft electric keys over slow major-ninth pads, without drums or vocals."""
    count = round(duration * RATE)
    music = numpy.zeros((count, 2), dtype=numpy.float32)
    rng = numpy.random.default_rng(7)
    beat = 60 / 75
    changes = [(50, 57, 61, 64, 66), (47, 54, 57, 61, 66),
               (43, 50, 54, 57, 62), (45, 52, 57, 59, 64)]

    def note(midi, start, length, level, pan, pad=False):
        """Add one enveloped note to the stereo score."""
        begin = round(start * RATE)
        length = min(round(length * RATE), count - begin)
        if length <= 0:
            return
        time = numpy.arange(length, dtype=numpy.float32) / RATE
        frequency = 440 * 2 ** ((midi - 69) / 12)
        if pad:
            envelope = (1 - numpy.exp(-time / 0.65)) * numpy.minimum((length / RATE - time) / 1.4, 1)
            tone = (numpy.sin(2 * numpy.pi * frequency * time)
                    + 0.23 * numpy.sin(2 * numpy.pi * frequency * 1.0015 * time)
                    + 0.08 * numpy.sin(4 * numpy.pi * frequency * time))
        else:
            envelope = (1 - numpy.exp(-time / 0.008)) * numpy.exp(-time / 0.85)
            tone = (numpy.sin(2 * numpy.pi * frequency * time)
                    + 0.24 * numpy.sin(4 * numpy.pi * frequency * time) * numpy.exp(-time / 0.3)
                    + 0.07 * numpy.sin(6 * numpy.pi * frequency * time) * numpy.exp(-time / 0.15))
        tone *= envelope * level
        music[begin:begin + length, 0] += tone * numpy.sqrt(1 - pan)
        music[begin:begin + length, 1] += tone * numpy.sqrt(pan)

    for bar in range(int(duration / (8 * beat)) + 1):
        start = bar * 8 * beat
        chord = changes[bar % len(changes)]
        for index, midi in enumerate(chord):
            note(midi, start, 8 * beat + 1.3, 0.015 if index else 0.022,
                 0.2 + index * 0.15, pad=True)
        for step, index in enumerate((0, 2, 4, 3, 1, 4, 2, 3)):
            note(chord[index] + 12, start + step * beat + 0.02, 2.7,
                 0.026 * rng.uniform(0.8, 1.05), rng.uniform(0.25, 0.75))
    for channel, delay in enumerate((0.23, 0.37)):
        shift = round(delay * RATE)
        music[shift:, channel] += 0.16 * music[:-shift, 1 - channel].copy()
    music *= 0.012 / float(numpy.sqrt(numpy.mean(music ** 2)))
    fade = min(count // 2, 3 * RATE)
    music[:fade] *= numpy.linspace(0, 1, fade)[:, None]
    music[-fade:] *= numpy.linspace(1, 0, fade)[:, None]
    assert numpy.isfinite(music).all() and numpy.max(numpy.abs(music)) < 0.25
    path = FOLDER / "build/music.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes((music * 32767).astype("<i2").tobytes())
    return path


def finish(rendered):
    """Mix music with narration and verify the final WebM before replacing it."""
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(rendered),
    ]))
    duration = float(info["format"]["duration"])
    assert 10 < duration < 600
    music = score(duration)
    output = FOLDER / "build/narrated.webm"
    filters = (f"[0:a]apad,atrim=duration={duration},asplit=2[voice][control];"
               "[1:a][control]sidechaincompress=threshold=0.08:ratio=2.5:attack=30:release=650[bed];"
               "[voice][bed]amix=inputs=2:duration=longest:normalize=0,"
               "loudnorm=I=-16:TP=-1.5:LRA=9[mix]")
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(rendered), "-i", str(music),
        "-i", str(FOLDER.parent / "results/Veyl.srt"), "-filter_complex", filters,
        "-map", "0:v:0", "-map", "[mix]", "-map", "2:0",
        "-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0", "-deadline", "realtime",
        "-cpu-used", "6", "-row-mt", "1", "-threads", "3", "-pix_fmt", "yuv420p",
        "-c:a", "libopus", "-b:a", "128k", "-ar", "48000", "-ac", "2", "-c:s", "webvtt",
        "-metadata:s:a:0", "language=eng", "-metadata:s:s:0", "language=eng",
        "-disposition:a:0", "default", "-disposition:s:0", "0", str(output),
    ], check=True)
    check(output)
    # Preserve the last deliverable until every spoken clip survives the final mix.
    output.replace(FOLDER.parent / "results/Veyl.webm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rendered", type=Path, help="Manim's rendered MP4, including its narration")
    finish(parser.parse_args().rendered)
