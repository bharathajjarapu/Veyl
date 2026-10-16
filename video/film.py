"""Render the narrated model explainer with retained weights and schematic training."""

import hashlib
import json
import os
import struct
import textwrap
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
os.environ["FONTCONFIG_FILE"] = str(FOLDER / "fonts.conf")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy
from scipy.special import erf
from narrate import read as narration
from manim import (
    Arrow, Brace, Circle, Create, Dot, FadeIn, FadeOut, Group,
    ImageMobject, LaggedStart, Line, ManimColor, ReplacementTransform,
    RoundedRectangle, Scene, SurroundingRectangle, Text, Transform, VGroup,
    DOWN, LEFT, RIGHT, UP, interpolate_color, register_font, tempconfig,
)

ROOT = FOLDER.parent
ASSETS = FOLDER / "assets"
BACK = "#FFFFFF"
INK = "#172335"
GRAY = "#5B687A"
LINE = "#E4EAF0"
MINT = "#087F70"
BLUE = "#385BC7"
GOLD = "#A36619"
RED = "#BE4054"
PIECES = ["[CLS]", "contact", "john", "smith", "today", ".", "[SEP]"]


def text(value, size=28, color=INK, mono=False):
    """Use a geometric typeface with readable diagram labels."""
    item = Text(value, font="JetBrains Mono" if mono else "Space Grotesk",
                font_size=size, color=color, weight="MEDIUM", disable_ligatures=True)
    assert item.width < 12.4, value
    return item


def words(item):
    """Reveal word slices without changing the line's typeset baseline."""
    assert len(item.chars) == len(item.original_text)
    assert item.original_text == " ".join(item.original_text.split())
    groups, start = [], 0
    for value in item.original_text.split():
        groups.append(VGroup(*item.chars[start:start + len(value)]))
        start += len(value) + 1
    return groups


def chip(value, color=GRAY, width=1.4):
    """Keep one token or state inside an outline."""
    frame = RoundedRectangle(width=width, height=0.65, corner_radius=0.09,
                             stroke_color=color, stroke_width=1.3,
                             fill_color=BACK, fill_opacity=1)
    label = text(value, 23, color)
    assert label.width < width - 0.12, value
    return VGroup(frame, label)


def arrow(start, end, color=GRAY):
    """Leave a visible gap at both connecting edges."""
    route = Arrow(start, end, buff=0.25, color=color, stroke_width=2,
                  tip_length=0.14, max_tip_length_to_length_ratio=0.4)
    assert numpy.linalg.norm(route.get_start() - start) >= 0.24
    assert numpy.linalg.norm(route.get_end() - end) >= 0.24
    return route


def stack(count, color, width=2.3):
    """Show one tile per BERT layer."""
    return VGroup(*[RoundedRectangle(width=width, height=0.18, corner_radius=0.035,
                                     stroke_color=color, stroke_width=1,
                                     fill_color=color, fill_opacity=0.08)
                    for index in range(count)]).arrange(UP, buff=0.06)


def vector(values, width=0.32):
    """Show real vector components with an ellipsis between the first and last."""
    cells = VGroup()
    visible = list(range(10)) + [None, None, None] + list(range(124, 128))
    for index, component in enumerate(visible):
        position = [0, 1.12 - index * 0.14, 0]
        if component is None:
            cells.add(Dot(position, radius=0.018, color=GRAY))
            continue
        value = float(values[component])
        color = interpolate_color(ManimColor(LINE), ManimColor(MINT if value >= 0 else BLUE),
                                  min(abs(value) / 3, 1))
        cell = RoundedRectangle(width=width, height=0.12, corner_radius=0.018,
                                stroke_width=0, fill_color=color, fill_opacity=1)
        cells.add(cell.move_to(position))
    return cells


def matrix(values):
    """Keep one visible column per token."""
    return VGroup(*[vector(row) for row in values]).arrange(RIGHT, buff=0.23)


def tokens():
    """Keep the same seven tokens in every inference illustration."""
    return VGroup(*[chip(piece, MINT if index in (2, 3) else GRAY)
                    for index, piece in enumerate(PIECES)]).arrange(RIGHT, buff=0.2)


def ribbon(values):
    """Show six early and two final components of a real vector."""
    cells = VGroup()
    for index, value in enumerate(numpy.r_[values[:6], values[-2:]]):
        color = interpolate_color(ManimColor(LINE), ManimColor(MINT if value >= 0 else BLUE),
                                  min(abs(float(value)) / 3, 1))
        cell = RoundedRectangle(width=0.22, height=0.28, corner_radius=0.025,
                                stroke_width=0, fill_color=color, fill_opacity=1)
        cells.add(cell.move_to([index * 0.27 + (0.2 if index >= 6 else 0), 0, 0]))
    dots = VGroup(*[Dot([1.55 + index * 0.05, 0, 0], radius=0.012, color=GRAY)
                    for index in range(3)])
    return VGroup(cells, dots).move_to([0, 0, 0])


def distribution(values, color):
    """Compare qualitative training scores on a fixed scale."""
    assert numpy.isfinite(values).all() and min(values) >= 0 and sum(values) <= 1.00001
    chart = VGroup()
    for index, (label, value) in enumerate(zip(("B-PERSON", "B-ORG", "O"), values)):
        height = 0.65 - index * 0.7
        name = text(label, 21, color).move_to([-2.05, height, 0], aligned_edge=LEFT)
        track = RoundedRectangle(width=1.55, height=0.23, corner_radius=0.03,
                                 stroke_width=0, fill_color=LINE, fill_opacity=1)
        track.move_to([0.7, height, 0])
        fill = RoundedRectangle(width=max(1.55 * value, 0.001), height=0.23, corner_radius=0.025,
                                stroke_width=0, fill_color=color, fill_opacity=0.85)
        fill.move_to(track.get_left(), aligned_edge=LEFT)
        chart.add(VGroup(name, track, fill))
    chart.add(text("…", 25, GRAY).move_to([-1.8, -1.4, 0]))
    return chart


def trace():
    """Compute the fixed FP32 example for animation without PyTorch."""
    raw = (ROOT / "weights/deep/model.safetensors").read_bytes()
    pins = json.loads((ROOT / "local/reports/files.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == pins["weights/deep/model.safetensors"]
    size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + size])
    config = json.loads((ROOT / "weights/deep/config.json").read_text())
    assert config["hidden_size"] == 128 and config["num_hidden_layers"] == 6
    assert config["num_attention_heads"] == 2 and config["intermediate_size"] == 512

    def weight(name):
        """Read one pinned float32 tensor."""
        entry = header[name]
        assert entry["dtype"] == "F32"
        start, end = entry["data_offsets"]
        return numpy.frombuffer(raw, dtype="<f4", count=(end - start) // 4,
                                offset=8 + size + start).reshape(entry["shape"])

    def linear(values, name):
        """Apply one retained linear projection."""
        return values @ weight(name + ".weight").T + weight(name + ".bias")

    def norm(values, name):
        """Use the checkpoint's LayerNorm epsilon and population variance."""
        values = (values - values.mean(-1, keepdims=True)) / numpy.sqrt(
            values.var(-1, keepdims=True) + config["layer_norm_eps"])
        return values * weight(name + ".weight") + weight(name + ".bias")

    vocab = json.loads((ROOT / "weights/tokenizer.json").read_text())["model"]["vocab"]
    hidden = weight("bert.embeddings.word_embeddings.weight")[[vocab[piece] for piece in PIECES]]
    hidden = hidden + weight("bert.embeddings.position_embeddings.weight")[:len(PIECES)]
    hidden = hidden + weight("bert.embeddings.token_type_embeddings.weight")[0]
    hidden = norm(hidden, "bert.embeddings.LayerNorm")
    states, maps, detail = [hidden.copy()], [], {}
    for layer in range(6):
        prefix = f"bert.encoder.layer.{layer}."
        query, keys, values = [linear(hidden, prefix + "attention.self." + name)
                               .reshape(len(PIECES), 2, 64).transpose(1, 0, 2)
                               for name in ("query", "key", "value")]
        scores = query @ keys.transpose(0, 2, 1) / 8
        probs = numpy.exp(scores - scores.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        assert numpy.allclose(probs.sum(-1), 1) and numpy.isfinite(probs).all()
        maps.append(probs.copy())
        mixed = probs @ values
        if layer == 0:
            detail = {"query": query.copy(), "keys": keys.copy(),
                      "values": values.copy(), "mixed": mixed.copy()}
            assert numpy.allclose(mixed[0, 2], probs[0, 2] @ values[0])
        context = mixed.transpose(1, 0, 2).reshape(len(PIECES), 128)
        hidden = norm(hidden + linear(context, prefix + "attention.output.dense"),
                      prefix + "attention.output.LayerNorm")
        inner = linear(hidden, prefix + "intermediate.dense")
        inner = 0.5 * inner * (1 + erf(inner / numpy.sqrt(2)))
        hidden = norm(hidden + linear(inner, prefix + "output.dense"),
                      prefix + "output.LayerNorm")
        states.append(hidden.copy())
    logits = linear(hidden, "classifier").copy()
    logits[:, 0] -= 1
    # This fixed trace illustrates Deep; the deployment also has the gated AGE head.
    assert logits.shape == (7, 57) and all(numpy.isfinite(state).all() for state in states)
    assert logits.argmax(-1)[1:6].tolist() == [0, 1, 2, 0, 0]
    return states, maps, logits, detail


class Veyl(Scene):
    def speak(self, key):
        """Start a measured clip after the matching diagram is visible."""
        path, duration, cues = narration(key)
        start = self.time
        assert not self.timeline or self.timeline[-1]["end"] <= start
        # Scene.add_sound skips cached animations, so write audio directly.
        self.renderer.file_writer.add_sound(str(path), time=start)
        self.voice_end = start + duration
        self.voice_cues = [start + cue["start"] for cue in cues]
        self.timeline.append({"key": key, "start": start, "end": self.voice_end})
        self.subtitles.extend({"text": cue["text"], "start": start + cue["start"],
                               "end": start + cue["end"]} for cue in cues)

    def cue(self, index):
        """Trigger a key visual change when its spoken sentence begins."""
        remaining = self.voice_cues[index] - self.time
        if remaining > 0:
            self.wait(remaining)

    def finish_voice(self):
        """Keep each diagram on screen until its narration has finished."""
        remaining = self.voice_end + 0.45 - self.time
        if remaining > 0:
            self.wait(remaining)

    def show(self, *items, duration=0.8):
        """Keep diagrams below the caption and inside the frame."""
        for item in items:
            assert item.get_left()[0] >= -6.7 and item.get_right()[0] <= 6.7
            assert item.get_bottom()[1] >= -3.65 and item.get_top()[1] <= 1.95
        self.play(LaggedStart(*[FadeIn(item, shift=UP * 0.05) for item in items],
                             lag_ratio=0.12), run_time=duration)

    def wipe(self):
        """Replace one complete frame without leaving text behind."""
        self.finish_voice()
        if self.mobjects:
            self.play(*[FadeOut(item) for item in self.mobjects], run_time=0.65)

    def stream(self, sentence, *items, keep=False):
        """Reveal typeset words above a new or continuing diagram."""
        assert len(sentence.split()) <= 16
        assert not {"i", "we", "our", "us"} & {word.lower().strip(".,:") for word in sentence.split()}
        if keep:
            self.finish_voice()
            self.play(FadeOut(self.caption), run_time=0.45)
        else:
            self.wipe()
        print(f"FRAME {sentence!r} {self.time:.2f}", flush=True)
        lines, width = [[]], 0
        for value in sentence.split():
            size = text(value, 32).width
            if lines[-1] and width + 0.16 + size > 12:
                lines.append([])
                width = 0
            lines[-1].append(value)
            width += size + 0.16
        assert len(lines) <= 2
        self.caption = VGroup(*[text(" ".join(line), 32) for line in lines])
        self.caption.arrange(DOWN, buff=0.17).move_to(UP * 3.35, aligned_edge=UP)
        assert self.caption.get_bottom()[1] > 2.2 and self.caption.get_right()[0] < 6.1
        assert abs(self.caption.get_center()[0]) < 1e-8
        if items:
            self.show(*items)
        self.speak(sentence)
        chunks = [word for line in self.caption for word in words(line)]
        self.play(LaggedStart(*[FadeIn(word, run_time=0.4) for word in chunks],
                             lag_ratio=0.65), run_time=1)
        self.remove(*chunks)
        self.add(self.caption)

    def brand(self, key):
        """Use the same minimal title card at both ends."""
        self.wipe()
        print(f"FRAME 'Veyl brand' {self.time:.2f}", flush=True)
        logo = text("VEYL", 86).move_to(UP * 0.8)
        privacy = text("Private text stays on your device.", 28).move_to(DOWN * 0.35)
        self.play(LaggedStart(*[FadeIn(letter, shift=UP * 0.05) for letter in logo],
                             lag_ratio=0.15), run_time=1.3)
        self.speak(key)
        chunks = words(privacy)
        self.play(LaggedStart(*[FadeIn(word, run_time=0.4) for word in chunks], lag_ratio=0.65))
        self.remove(*chunks)
        self.add(privacy)
        self.finish_voice()

    def construct(self):
        """Render the film with the bundled heading font."""
        self.voice_end = -0.45
        self.timeline, self.subtitles = [], []
        (FOLDER / "build").mkdir(exist_ok=True)
        with register_font(str(ASSETS / "grotesk.ttf")):
            self.render_model()
        self.finish_voice()
        assert len(self.timeline) == 19
        assert self.timeline[-1]["end"] <= self.time
        (FOLDER / "build/timing.json").write_text(json.dumps(self.timeline, indent=2) + "\n")

        def timestamp(seconds):
            """Format seconds as an SRT timestamp."""
            milliseconds = round(seconds * 1000)
            hours, milliseconds = divmod(milliseconds, 3600000)
            minutes, milliseconds = divmod(milliseconds, 60000)
            seconds, milliseconds = divmod(milliseconds, 1000)
            return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

        blocks = [f"{index}\n{timestamp(cue['start'])} --> {timestamp(cue['end'])}\n"
                  + textwrap.fill(cue["text"], width=64)
                  for index, cue in enumerate(self.subtitles, 1)]
        (ROOT / "results/Veyl.srt").write_text("\n\n".join(blocks) + "\n")

    def render_model(self):
        """Pair concise narration with each model illustration."""
        self.camera.background_color = BACK
        states, maps, logits, detail = trace()
        gate = json.loads((ROOT / "local/reports/gate.json").read_text())
        assert gate["veyl_model"]["parameters"] == 5_170_107
        assert gate["veyl_model"]["onnx_bytes_actual"] == 5_426_280

        self.brand("intro")

        panels = Group()
        for position in (-2.7, 2.7):
            frame = RoundedRectangle(width=3.9, height=3.8, corner_radius=0.16,
                                     stroke_color=LINE, stroke_width=1,
                                     fill_color=BACK, fill_opacity=1)
            panels.add(frame.move_to([position, -0.3, 0]))
        nvidia = ImageMobject(str(ASSETS / "nvidia.png")).set(width=3.55).move_to([-2.7, 0.6, 0])
        privacy = ImageMobject(str(ASSETS / "privacy.png")).set(height=1.15).move_to([2.7, 0.6, 0])
        labels = VGroup(text("Nemotron-PII", 27, MINT).move_to([-2.7, -0.88, 0]),
                        text("OpenPII 1M", 27, BLUE).move_to([2.7, -0.88, 0]),
                        text("Base + AGE tuning", 21, GRAY).move_to([-2.7, -1.6, 0]),
                        text("AGE tuning", 21, GRAY).move_to([2.7, -1.6, 0]),
                        text("AI4Privacy", 21).move_to([2.7, -0.32, 0]))
        self.stream("Veyl is trained on the following datasets.",
                    panels, nvidia, privacy, labels)

        pieces = VGroup(*[chip(word, MINT if index in (1, 2) else INK, width=2.2)
                          for index, word in enumerate(("Contact", "John", "Smith", "today."))])
        pieces.arrange(RIGHT, buff=0.65).move_to(UP * 0.8)
        self.stream("The dataset marks where private information begins and continues.", pieces)
        labels = VGroup()
        for piece, tag in zip(pieces, ("O", "B-PERSON", "I-PERSON", "O")):
            label = text(tag, 23, MINT if "PERSON" in tag else GRAY, mono=True)
            label.next_to(piece, DOWN, buff=0.65)
            labels.add(label)
            self.play(Create(arrow(piece.get_bottom(), label.get_top())), FadeIn(label), run_time=0.55)
        legend = VGroup(text("B = begins", 24, MINT), text("I = continues", 24, MINT),
                        text("O = outside", 24, GRAY)).arrange(RIGHT, buff=0.8).move_to(DOWN * 2.1)
        self.show(legend)

        frame = RoundedRectangle(width=2.2, height=2.6, corner_radius=0.15,
                                 stroke_color=BLUE, stroke_width=1.3,
                                 fill_color=BACK, fill_opacity=1)
        body = stack(12, BLUE).scale(0.5).move_to(UP * 0.1)
        title = text("Pretrained BERT", 26, BLUE).move_to(UP * 1.72)
        size = text("12 × 768", 21, BLUE, mono=True).move_to(DOWN * 1.0)
        assert title.get_bottom()[1] - frame.get_top()[1] > 0.2
        assert body.get_bottom()[1] - size.get_top()[1] > 0.15
        teacher = VGroup(frame, body, title, size)
        sample = chip("Contact John Smith today.", INK, width=4.6).scale_to_fit_width(4)
        sample[0].stretch_to_fit_height(0.65)
        sample.move_to([-4, 0, 0])

        prediction = chip("O", RED, width=4).move_to([4, 0, 0])
        labels = words(text("Input text Prediction for John", 23))
        inputname, outputname = VGroup(*labels[:2]), VGroup(*labels[2:])
        headings = VGroup(inputname, outputname)
        inputname.set_color(GRAY).shift(RIGHT * (-4 - inputname.get_center()[0]))
        outputname.set_color(BLUE).shift(RIGHT * (4 - outputname.get_center()[0]))
        headings.shift(UP * (0.85 - headings.get_center()[1]))
        target = chip("B-PERSON", GOLD, width=4).move_to([4, -1.6, 0])
        truth = text("Dataset label", 23, GOLD).move_to([4, -2.25, 0])
        self.stream("First, the teacher learns the correct token labels.",
                    teacher, sample, inputname, prediction, outputname, target, truth)
        routes = VGroup(arrow(sample.get_right(), frame.get_left(), color=BLUE),
                        arrow(frame.get_right(), prediction.get_left(), color=BLUE))
        assert numpy.isclose(routes[0].get_length(), routes[1].get_length())
        self.play(LaggedStart(*[Create(route) for route in routes], lag_ratio=0.3), run_time=0.8)
        for opacity in (0.16, 0.3, 0.44):
            self.play(LaggedStart(*[layer.animate.set_fill(BLUE, opacity=opacity) for layer in body],
                                 lag_ratio=0.07), run_time=0.8)
        correct = chip("B-PERSON", BLUE, width=4).move_to(prediction)
        self.cue(1)
        self.play(Transform(prediction, correct), target[0].animate.set_fill(GOLD, opacity=0.1), run_time=0.8)

        source = tokens().scale(0.64).move_to(UP * 1.25)
        teacher = stack(12, BLUE).scale(0.7).move_to([-3.15, -1.1, 0])
        student = stack(6, MINT).scale(0.7).move_to([3.15, -1.1, 0])
        names = VGroup(text("Frozen teacher", 28, BLUE).move_to([-3.15, -2.6, 0]),
                       text("Veyl", 28, MINT).move_to([3.15, -2.6, 0]),
                       text("12 × 768", 22, BLUE).move_to([-3.15, -3.1, 0]),
                       text("6 × 128 · 2 heads", 22, MINT).move_to([3.15, -3.1, 0]))
        self.stream("Both models read the same tokens. The teacher stays frozen.", source)
        first = source.copy().scale(0.8).move_to([-3.15, 0.55, 0])
        second = source.copy().scale(0.8).move_to([3.15, 0.55, 0])
        self.play(ReplacementTransform(source.copy(), first),
                  ReplacementTransform(source.copy(), second), FadeOut(source), run_time=1.1)
        self.show(teacher, student, names)
        self.play(LaggedStart(*[layer.animate.set_fill(BLUE, opacity=0.25) for layer in teacher],
                             lag_ratio=0.06),
                  LaggedStart(*[layer.animate.set_fill(MINT, opacity=0.25) for layer in student],
                             lag_ratio=0.12), run_time=1.3)

        dataset = distribution([1, 0, 0], GOLD).move_to([-4.3, -0.1, 0])
        target = distribution([0.7, 0.15, 0.1], BLUE).move_to([0, -0.1, 0])
        current = distribution([0.35, 0.35, 0.2], MINT).move_to([4.3, -0.1, 0])
        titles = VGroup(*words(text("Dataset Teacher Veyl", 28)))
        for title, position, color in zip(titles, (-4.3, 0, 4.3), (GOLD, BLUE, MINT)):
            title.set_color(color).shift(RIGHT * (position - title.get_center()[0]))
        titles.shift(UP * (1.65 - titles.get_center()[1]))
        for title, chart in zip(titles, (dataset, target, current)):
            assert title.get_bottom()[1] - chart.get_top()[1] > 0.35
        comparison = Brace(VGroup(dataset, target, current), DOWN, buff=0.2, color=GRAY)
        loss = chip("Combined loss", GRAY, width=3.1).move_to([0, -2.4, 0])
        self.stream("Veyl learns from dataset labels and teacher scores.",
                    dataset, target, current, titles, loss)
        self.play(Create(comparison), run_time=0.6)

        self.stream("Veyl updates its weights using both signals.", keep=True)
        self.cue(1)
        for fraction in (0.35, 0.7, 1.0):
            values = (1 - fraction) * numpy.array([0.35, 0.35, 0.2]) + fraction * numpy.array([0.86, 0.05, 0.04])
            scores = distribution(values, MINT).move_to(current)
            self.play(Transform(current, scores), run_time=1.2)

        base = stack(6, BLUE, width=1.6).move_to([-3.5, 0.5, 0])
        head = chip("AGE head", GOLD, width=2.0).move_to([2, -1.8, 0])
        name = text("258 parameters", 23, GOLD).next_to(head, DOWN, buff=0.3)
        source = text("Frozen BERT", 27, BLUE).move_to([-3.5, 1.6, 0])
        switch = chip("O", MINT, width=1.0).move_to([0, 0.5, 0])
        titles = VGroup(text("Base", 23, BLUE).move_to([0, 1.3, 0]),
                        text("Final", 23, BLUE).move_to([4, 1.3, 0]))
        output = chip("AGE", GOLD).move_to([4.0, 0.5, 0])
        route = arrow(base.get_right(), switch.get_left(), color=BLUE)
        residual = arrow(head.get_top(), numpy.array([2, 0.5, 0]), color=GOLD)
        result = arrow(switch.get_right(), output.get_left(), color=GOLD)
        self.stream("The AGE head can relabel O tokens. Existing entity predictions stay unchanged.",
                    base, head, name, source, switch, titles, output, route, residual, result)
        self.wait(1.5)
        person = chip("PERSON", BLUE, width=1.8).move_to(switch)
        keep = chip("PERSON", BLUE, width=1.8).move_to(output)
        self.cue(2)
        self.play(Transform(switch, person), Transform(output, keep),
                  residual.animate.set_opacity(0.15), head.animate.set_opacity(0.4), name.animate.set_opacity(0.4),
                  Transform(route, arrow(base.get_right(), person.get_left(), color=BLUE)),
                  Transform(result, arrow(person.get_right(), keep.get_left(), color=BLUE)), run_time=1)
        self.wait(1.5)

        pieces = tokens().move_to(UP * 1.1)
        columns = VGroup(*[vector(row).move_to([-4.8 + index * 1.6, -0.9, 0])
                           for index, row in enumerate(states[0])])
        self.stream("Every token gets a 128-dimensional vector.", pieces)
        self.play(LaggedStart(*[FadeIn(column, shift=DOWN * 0.15) for column in columns],
                             lag_ratio=0.12), run_time=1.5)
        parts = VGroup(*[chip(value, GRAY, width=2.4) for value in ("Token", "Position", "Segment")])
        parts.arrange(RIGHT, buff=0.7).move_to(DOWN * 2.9)
        plus = VGroup(*[text("+", 25, GRAY).move_to((parts[index].get_right() +
                        parts[index + 1].get_left()) / 2) for index in range(2)])
        self.show(parts, plus)

        john = chip("john", MINT, width=1.8).move_to([-4.3, -0.25, 0])
        value = ribbon(states[0][2]).next_to(john, DOWN, buff=0.4)
        self.stream("Each token produces a query, a key and a value.", john, value)
        roles = (("Query", "What to match", "query", MINT),
                 ("Key", "How to match", "keys", BLUE),
                 ("Value", "Information to share", "values", GOLD))
        for index, (name, role, field, color) in enumerate(roles):
            self.cue(index + 1)
            height = 1.1 - index * 1.35
            label = chip(name, color, width=1.5).move_to([-0.75, height, 0])
            numbers = ribbon(detail[field][0, 2]).move_to([1.3, height, 0])
            explanation = text(role, 23, color).move_to([4.4, height, 0])
            route = arrow(john.get_right(), label.get_left(), color=color)
            self.play(Create(route), FadeIn(label), FadeIn(numbers), FadeIn(explanation), run_time=0.9)
        size = text("64 dimensions per head", 23, GRAY).move_to(DOWN * 2.8)
        self.show(size)

        pieces = tokens().move_to(UP * 1.15)
        focus = SurroundingRectangle(pieces[2], color=MINT, buff=0.08, corner_radius=0.08)
        self.stream("John's query scores each key. Softmax turns those scores into weights.", pieces)
        self.play(Create(focus), run_time=0.5)
        self.cue(1)
        weights = maps[0][0, 2]
        baseline = -1.65
        chart = VGroup()
        for index, weight in enumerate(weights):
            height = 2.1 * float(weight) + 0.018
            bar = RoundedRectangle(width=0.72, height=height, corner_radius=0.008,
                                   stroke_width=0, fill_color=MINT if index == 3 else BLUE,
                                   fill_opacity=0.85)
            bar.move_to([-4.8 + index * 1.6, baseline, 0], aligned_edge=DOWN)
            label = text(f"{weight:.1%}", 23, MINT if index == 3 else GRAY, mono=True)
            label.next_to(bar, UP, buff=0.18)
            chart.add(VGroup(bar, label))
        self.play(LaggedStart(*[FadeIn(item, shift=UP * 0.1) for item in chart], lag_ratio=0.12), run_time=1.5)
        self.play(pieces[3][0].animate.set_fill(MINT, opacity=0.12), run_time=0.6)
        self.cue(2)
        self.show(text("Layer 1 · head 1", 22, GRAY).move_to(DOWN * 2.7))

        pieces = tokens().move_to(UP * 1.2)
        values = VGroup(*[ribbon(detail["values"][0, index]).scale(0.56)
                          .move_to([-4.8 + index * 1.6, 0.15, 0]) for index in range(7)])
        self.stream("These weights scale value vectors. Their sum becomes John's context.", pieces, values)
        weighted = VGroup()
        for index, weight in enumerate(weights):
            part = ribbon(detail["values"][0, index] * weight).scale(0.56).move_to(values[index])
            weighted.add(part)
        amounts = VGroup(*[text(f"× {weight:.1%}", 20, MINT if index == 3 else GRAY)
                           .move_to([-4.8 + index * 1.6, -0.55, 0]) for index, weight in enumerate(weights)])
        self.show(amounts)
        self.play(Transform(values, weighted), run_time=1)
        self.cue(1)
        result = ribbon(detail["mixed"][0, 2]).move_to([0, -2.0, 0])
        sumlabel = text("Weighted sum", 24, MINT).move_to([0, -1.25, 0])
        name = text("Context for john · 64 dimensions", 23, MINT).move_to([0, -2.8, 0])
        self.show(sumlabel)
        self.play(ReplacementTransform(values.copy(), result), run_time=1.2)
        self.show(name)

        heads = VGroup()
        for head, color, position in [(0, MINT, -3.15), (1, BLUE, 3.15)]:
            label = text(f"Head {head + 1}", 28, color).move_to([position, 1.35, 0])
            chart = VGroup()
            for index, weight in enumerate(maps[0][head, 2]):
                bar = RoundedRectangle(width=0.28, height=1.5 * float(weight) + 0.025,
                                       corner_radius=0.008, stroke_width=0,
                                       fill_color=color, fill_opacity=0.8)
                bar.move_to([position - 1.15 + index * 0.38, -0.25, 0], aligned_edge=DOWN)
                chart.add(bar)
            numbers = ribbon(detail["mixed"][head, 2]).move_to([position, -0.9, 0])
            size = text("64 dimensions", 22, color).move_to([position, -1.5, 0])
            heads.add(VGroup(label, chart, numbers, size))
        self.stream("Two heads combine 64 dimensions each into 128 dimensions.", heads)
        joined = VGroup(ribbon(detail["mixed"][0, 2]), ribbon(detail["mixed"][1, 2]))
        joined.arrange(RIGHT, buff=0.16).move_to([0, -2.6, 0])
        self.cue(2)
        self.play(ReplacementTransform(heads[0][2].copy(), joined[0]),
                  ReplacementTransform(heads[1][2].copy(), joined[1]), run_time=1.2)
        size = text("128 dimensions → output projection", 23, GRAY).move_to(DOWN * 3.3)
        self.show(size)

        blocks = VGroup(*[chip(value, color, width=2.35) for value, color in
                            [("Attention", MINT), ("Add + norm", GRAY),
                             ("Feed-forward", BLUE), ("Add + norm", GRAY)]])
        blocks.arrange(RIGHT, buff=0.85)
        blocks[2:].shift(RIGHT * 0.55)
        blocks.move_to([0, 0.2, 0])
        branch = (blocks[1].get_right() + blocks[2].get_left()) / 2
        routes = VGroup(*[arrow(first.get_right(), second.get_left())
                          for first, second in zip(blocks, blocks[1:])])
        assert routes[1].get_start()[0] + 0.3 < branch[0] < routes[1].get_end()[0] - 0.3
        detail = VGroup(text("2 heads × 64", 23, MINT).next_to(blocks[0], DOWN, buff=0.5),
                        text("128 → 512 → 128", 21, BLUE, mono=True).next_to(blocks[2], DOWN, buff=0.5))
        self.stream("Each BERT layer combines attention, a feed-forward network and residual connections.", blocks)
        self.play(LaggedStart(*[Create(route) for route in routes], lag_ratio=0.3), run_time=1)
        self.show(detail)
        self.cue(1)
        for first, second, color in [(blocks[0], blocks[1], MINT), (blocks[2], blocks[3], BLUE)]:
            position = first.get_left()[0] - 0.45 if first is blocks[0] else branch[0]
            start = numpy.array([position, first.get_center()[1], 0])
            bend = numpy.array([position, 1.5, 0])
            end = numpy.array([second.get_top()[0], 1.5, 0])
            skip = VGroup(Line(start, bend, color=color, stroke_width=2),
                          Line(bend, end, color=color, stroke_width=2),
                          arrow(end + UP * 0.25, second.get_top(), color=color))
            if first is blocks[0]:
                skip.add(Line(start, first.get_left() + LEFT * 0.25, color=color, stroke_width=2))
            assert start[0] < first.get_left()[0]
            self.play(Create(skip), first[0].animate.set_fill(color, opacity=0.12), run_time=0.8)

        layers = VGroup(*[RoundedRectangle(width=0.7, height=0.05, corner_radius=0.025,
                                           stroke_width=0, fill_color=LINE, fill_opacity=1)
                          for index in range(6)]).arrange(RIGHT, buff=0.2).move_to(UP * 0.95)
        pieces = tokens().move_to(UP * 0.1)
        hidden = matrix(states[0]).arrange(RIGHT, buff=1.28).stretch_to_fit_height(1.85).move_to(DOWN * 1.55)
        stage = text("Input vectors", 26, GRAY).move_to(UP * 1.65)
        note = text("Each layer updates every token vector.", 22, GRAY).move_to(DOWN * 3.3)
        assert stage.get_bottom()[1] - layers.get_top()[1] > 0.45
        assert layers.get_bottom()[1] - pieces.get_top()[1] > 0.45
        assert pieces.get_bottom()[1] - hidden.get_top()[1] > 0.35
        self.stream("Six BERT layers run in sequence.", layers, pieces, hidden, stage, note)
        for index, layer in enumerate(layers):
            updated = matrix(states[index + 1]).arrange(RIGHT, buff=1.28).stretch_to_fit_height(1.85).move_to(hidden)
            label = text(f"Layer {index + 1} of 6", 26, MINT).move_to(stage)
            self.remove(stage)
            self.add(label)
            stage = label
            self.play(layer.animate.set_fill(MINT, opacity=1),
                      Transform(hidden, updated), run_time=1.2)
            self.play(layer.animate.set_fill(MINT, opacity=0.35), run_time=0.35)

        hidden = matrix(states[-1]).move_to([-3.5, -0.3, 0])
        nodes = VGroup(*[Circle(radius=0.065, stroke_width=0, fill_color=GRAY, fill_opacity=0.35)
                         .move_to([0.2 + index % 19 * 0.27, 0.4 - index // 19 * 0.38, 0])
                         for index in range(57)])
        count = text("57 labels", 28, GRAY).next_to(nodes, UP, buff=0.4)
        project = arrow(hidden.get_right(), nodes.get_left(), color=GRAY)
        pieces = VGroup(chip("John", MINT, width=1.8), chip("Smith", MINT, width=1.8))
        pieces.arrange(RIGHT, buff=0.8).move_to([1.9, -2.0, 0])
        self.stream("BIO labels merge into spans for redaction.", hidden, nodes, count, project, pieces)
        self.cue(1)
        for column, word, tag in [(2, pieces[0], "B-PERSON"), (3, pieces[1], "I-PERSON")]:
            selected = SurroundingRectangle(hidden[column], color=MINT, buff=0.07, stroke_width=1.5)
            self.play(Create(selected), run_time=0.55)
            probs = numpy.exp(logits[column] - logits[column].max())
            probs /= probs.sum()
            self.play(*[node.animate.set_fill(MINT, opacity=0.08 + 0.92 * float(prob))
                        for node, prob in zip(nodes, probs)], run_time=0.6)
            label = text(tag, 21, MINT, mono=True).next_to(word, UP, buff=0.3)
            self.show(label)
            self.play(FadeOut(selected), run_time=0.45)
        span = Brace(pieces, DOWN, buff=0.2, color=MINT)
        entity = text("PERSON span", 23, MINT).next_to(span, DOWN, buff=0.2)
        self.cue(2)
        self.play(Create(span), run_time=0.6)
        self.show(entity)

        pins = json.loads((ROOT / "local/reports/files.json").read_text())
        report = (ROOT / "local/reports/piimb.json").read_bytes()
        assert hashlib.sha256(report).hexdigest() == pins["local/reports/piimb.json"]
        score = json.loads(report)
        assert score["prediction_meta"]["model_sha256"] == gate["veyl_model"]["sha256"]
        assert score["rows"] == 121409 and round(score["unweighted_task_f2_mean"] * 100, 2) == 88.03
        report = (ROOT / "local/reports/fp32.json").read_bytes()
        assert hashlib.sha256(report).hexdigest() == pins["local/reports/fp32.json"]
        full = json.loads(report)
        assert full["rows"] == score["rows"] and full["source_sha256"] == score["source_sha256"]
        assert full["prediction_meta"]["input_sha256"]["weights/deep/model.safetensors"] == pins["weights/deep/model.safetensors"]
        assert full["prediction_meta"]["runtime"]["dtype"] == "float32"
        native = sum((ROOT / name).stat().st_size for name in ("weights/deep/model.safetensors", "weights/head.pt"))
        assert native == 20_694_105

        titles = VGroup(text("FP32", 30, BLUE).move_to([-3.2, 1.7, 0]),
                        text("INT8", 30, MINT).move_to([3.2, 1.7, 0]))
        runtime = VGroup(text("PyTorch", 20, GRAY).move_to([-3.2, 1.05, 0]),
                         text("ONNX · WASM", 20, GRAY).move_to([3.2, 1.05, 0]))
        floats = VGroup(*[RoundedRectangle(width=0.23, height=0.23, corner_radius=0.025,
                                           stroke_width=0, fill_color=BLUE, fill_opacity=0.7)
                          for index in range(64)]).arrange_in_grid(rows=8, buff=0.06).set(height=1.4).move_to([-3.2, -0.15, 0])
        packed = VGroup(*[RoundedRectangle(width=0.115, height=0.115, corner_radius=0.015,
                                           stroke_width=0, fill_color=MINT, fill_opacity=0.85)
                          for index in range(64)]).arrange_in_grid(rows=8, buff=0.03).set(height=0.7).move_to([3.2, -0.15, 0])
        assert len(floats) == len(packed) == 64
        assert numpy.isclose((floats.width * floats.height) / (packed.width * packed.height), 4)
        size = VGroup(text(f"{native / 1e6:.2f} MB", 30, BLUE, mono=True).move_to([-3.2, -1.5, 0]),
                      text("5.43 MB", 30, MINT, mono=True).move_to([3.2, -1.5, 0]))
        scores = VGroup(text(f"{full['unweighted_task_f2_mean']:.2%}", 31, BLUE).move_to([-3.2, -2.45, 0]),
                        text(f"{score['unweighted_task_f2_mean']:.2%}", 31, MINT).move_to([3.2, -2.45, 0]))
        scope = text("Masking F2 · Local PIIMB English", 21, GRAY).move_to(DOWN * 3.35)
        assert runtime.get_bottom()[1] - floats.get_top()[1] > 0.35
        assert floats.get_bottom()[1] - size.get_top()[1] > 0.4
        assert size.get_bottom()[1] - scores.get_top()[1] > 0.5
        assert scores.get_bottom()[1] - scope.get_top()[1] > 0.5
        assert titles.get_bottom()[1] - runtime.get_top()[1] > 0.3
        self.stream("Same 5.17M parameters. Less storage.", titles, runtime, floats, size, scores, scope)
        self.play(ReplacementTransform(floats.copy(), packed), run_time=2.4)
        self.stream("Measured masking scores use different runtime and tokenizer paths.", keep=True)
        self.finish_voice()
        self.brand("outro")


if __name__ == "__main__":
    with tempconfig({"quality": "high_quality", "frame_rate": 60,
                     "media_dir": str(FOLDER / "build"), "output_file": "film",
                     "max_files_cached": 300, "progress_bar": "none"}):
        scene = Veyl()
        scene.render()
        rendered = Path(scene.renderer.file_writer.movie_file_path)
        from soundtrack import finish
        finish(rendered)
