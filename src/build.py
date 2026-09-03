"""Rebuild the selected VEYL graph from its retained AGE graph."""

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = "6c84a89021370c6b523babca91b019401638254b9de8aab4e192266fc9da1e6f"
SOURCE = "10d2a440236cdcaa50342be778dbd1f2d5fc177fefe448a827a0501cb8c265c6"


def digest(path):
    """Hash a retained file without changing it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch(source, target):
    """Allow the AGE residual only where the frozen classifier predicts O."""
    import numpy
    import onnx
    from onnx import helper, numpy_helper

    graph = onnx.load(str(source))
    soft = "/Softplus_output_0"
    base = "/classifier/Add_output_0"
    adds = [
        node
        for node in graph.graph.node
        if node.op_type == "Add" and soft in node.input
    ]
    if len(adds) != 1 or not any(base in node.output for node in graph.graph.node):
        raise ValueError("Expected frozen classifier and unique AGE residual")
    if any(
        value.name in {"core_zero_i64", "core_zero_f32"}
        for value in graph.graph.initializer
    ):
        raise ValueError("AGE gate already exists")
    graph.graph.initializer.extend(
        [
            numpy_helper.from_array(
                numpy.asarray(0, dtype=numpy.int64), "core_zero_i64"
            ),
            numpy_helper.from_array(
                numpy.asarray(0, dtype=numpy.float32), "core_zero_f32"
            ),
        ]
    )
    nodes = [
        helper.make_node(
            "ArgMax",
            [base],
            ["core_base_argmax"],
            name="core/ArgMaxBase",
            axis=-1,
            keepdims=1,
            select_last_index=0,
        ),
        helper.make_node(
            "Equal",
            ["core_base_argmax", "core_zero_i64"],
            ["core_base_is_o"],
            name="core/BaseIsO",
        ),
        helper.make_node(
            "Where",
            ["core_base_is_o", soft, "core_zero_f32"],
            ["core_gated_age_delta"],
            name="core/GateAgeResidual",
        ),
    ]
    residual = adds[0]
    residual.input[:] = [
        "core_gated_age_delta" if value == soft else value for value in residual.input
    ]
    index = list(graph.graph.node).index(residual)
    for offset, node in enumerate(nodes):
        graph.graph.node.insert(index + offset, node)
    props = {value.key: value.value for value in graph.metadata_props}
    # Historical graph names and metadata stay byte-identical to the accepted artifact.
    props.update(
        {
            "core_policy": "AGE residual is zero wherever frozen 57-way argmax is not O",
            "core_source_sha256": digest(source),
            "core_age_off": "maps residual-created AGE labels back to frozen O",
        }
    )
    del graph.metadata_props[:]
    for name, value in props.items():
        graph.metadata_props.add(key=name, value=value)
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, str(target))


def build(target):
    """Verify a rebuilt bundle before adding any missing destination assets."""
    source = ROOT / "weights/source.onnx"
    pins = json.loads((ROOT / "weights/sha.json").read_text())
    if digest(source) != SOURCE or pins.get("model.onnx") != PIN:
        raise ValueError("Selected source or graph pin changed")
    if set(pins) != {"model.onnx", "tokenizer.json", "labels.json"}:
        raise ValueError("Incomplete or unexpected bundle pins")
    for name, expected in pins.items():
        if (target / name).exists() and digest(target / name) != expected:
            raise ValueError(
                "Refusing to overwrite different asset: " + str(target / name)
            )
    if (target / "sha.json").exists() and json.loads(
        (target / "sha.json").read_text()
    ) != pins:
        raise ValueError("Destination manifest differs")
    with tempfile.TemporaryDirectory(prefix="veyl-") as scratch:
        stage = Path(scratch)
        patch(source, stage / "model.onnx")
        for name in ("tokenizer.json", "labels.json"):
            shutil.copy2(ROOT / "weights" / name, stage / name)
        for name, expected in pins.items():
            if digest(stage / name) != expected:
                raise ValueError("Rebuilt asset differs: " + name)
        target.mkdir(parents=True, exist_ok=True)
        for name in pins:
            if not (target / name).exists():
                shutil.copy2(stage / name, target / name)
        if not (target / "sha.json").exists():
            (target / "sha.json").write_text(json.dumps(pins, indent=2) + "\n")
    return {
        "source": SOURCE,
        "assets": pins,
        "bytes": (target / "model.onnx").stat().st_size,
    }


def check():
    """Check exact reconstruction and preservation of a different destination."""
    with tempfile.TemporaryDirectory(prefix="veyl-check-") as scratch:
        target = Path(scratch) / "weights"
        build(target)
        (target / "labels.json").write_text("corrupted fixture")
        try:
            build(target)
        except ValueError as error:
            assert "Refusing to overwrite" in str(error)
        else:
            raise AssertionError("Different destination was overwritten")
        assert digest(target / "model.onnx") == PIN
    print("Exact graph reconstruction and destination preservation passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        check()
    else:
        if len(sys.argv) > 2:
            raise SystemExit("Usage: python src/build.py [folder|--check]")
        print(
            json.dumps(
                build(Path(sys.argv[1]) if len(sys.argv) == 2 else ROOT / "weights"),
                indent=2,
            )
        )
