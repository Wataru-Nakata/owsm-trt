"""Reusing decoder engines must retain their bytes/metadata and avoid unused export work."""

import json
from types import SimpleNamespace

import pytest
from test_model import TINY

from owsm_trt import build


@pytest.mark.parametrize("reuse,skip", [(True, True), (True, False), (False, True)])
def test_export_routing_keeps_copied_engines_and_profiles(tmp_path, monkeypatch, reuse, skip):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    flags = dict(decoder=True, verify=False)
    (source / "build.json").write_text(json.dumps(dict(attn_plugin=flags)))
    for name in ("decoder", "verify"):
        (source / f"{name}.engine").write_bytes(f"original-{name}".encode())
    calls = []

    def encoder(model, directory, opts):
        calls.append("encoder")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "encoder.onnx"
        path.write_bytes(b"encoder-graph")
        return path

    def both(model, directory, opts):
        if reuse and skip:
            pytest.fail("exported an unused decoder")
        calls.append("decoder")
        path = directory / "decoder.onnx"
        encoded = encoder(model, directory, opts)
        path.write_bytes(b"decoder-graph")
        return encoded, path

    def compile_graph(path, output, *args):
        output.write_bytes(path.read_bytes())

    monkeypatch.setattr(build, "export_encoder_onnx", encoder)
    monkeypatch.setattr(build, "export_onnx", both)
    monkeypatch.setattr(build, "_build", compile_graph)
    opts = build.BuildOptions(attn_plugin=False, attn_plugin_steps=False)
    build.build_engines(SimpleNamespace(cfg=TINY), target, opts, keep_onnx=True,
                        decoders_from=source if reuse else None, skip_decoder_export=skip)
    assert calls == (["encoder"] if reuse and skip else ["decoder", "encoder"])
    assert (target / "encoder.engine").read_bytes() == b"encoder-graph"
    meta = json.loads((target / "build.json").read_text())
    for name in ("decoder", "verify"):
        expected = (source / f"{name}.engine").read_bytes() if reuse else b"decoder-graph"
        assert (target / f"{name}.engine").read_bytes() == expected
        assert meta["profiles"][name]
    if reuse:
        assert meta["attn_plugin"] == flags
        assert meta["decoders_from"] == source.name
