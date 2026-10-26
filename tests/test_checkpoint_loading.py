"""Small storage-format/alias tests; no model or audio loading."""

import hashlib
import io
import mmap
import pickle

import pytest
import torch

from owsm_trt.modeling import load_checkpoint


def _state():
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    return {"weight": x, "view": x[:, ::2], "ids": torch.tensor([3, 9], dtype=torch.int64)}


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("as_string", [False, True])
def test_checkpoint_values_aliases_and_source_unchanged(tmp_path, legacy, as_string):
    path = tmp_path / "model.pth"
    torch.save(_state(), path, _use_new_zipfile_serialization=not legacy)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    eager = load_checkpoint(path)
    mapped = load_checkpoint(str(path) if as_string else path, memory_map=True)
    assert mapped.keys() == eager.keys()
    for key in eager:
        assert torch.equal(mapped[key], eager[key])
        assert mapped[key].stride() == eager[key].stride()
    assert mapped["view"].untyped_storage().data_ptr() == mapped["weight"].untyped_storage().data_ptr()
    mapped["weight"][0, 0] = -7
    assert mapped["view"][0, 0] == -7
    assert eager["weight"][0, 0] == 0
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum
    assert load_checkpoint(path)["weight"][0, 0] == 0


def test_file_like_input_keeps_ordinary_load():
    buffer = io.BytesIO()
    state = _state()
    torch.save(state, buffer)
    buffer.seek(0)
    result = load_checkpoint(buffer, memory_map=True)
    for name in state:
        assert torch.equal(state[name], result[name])


@pytest.mark.skipif(not hasattr(mmap, "MAP_PRIVATE"), reason="POSIX mapping options")
def test_zip_checkpoint_really_selects_private_mapping(tmp_path, monkeypatch):
    path = tmp_path / "model.pth"
    torch.save(_state(), path)
    original = torch.load
    calls = []

    def checked(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked)
    with torch.serialization.set_default_mmap_options(mmap.MAP_PRIVATE):
        load_checkpoint(path, memory_map=True)
    assert calls[0]["mmap"] is True and calls[0]["weights_only"] is True


@pytest.mark.skipif(not hasattr(mmap, "MAP_SHARED"), reason="POSIX mapping options")
def test_shared_mapping_setting_falls_back_without_modifying_file(tmp_path, monkeypatch):
    path = tmp_path / "model.pth"
    torch.save(_state(), path)
    original = torch.load
    calls = []

    def checked(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked)
    with torch.serialization.set_default_mmap_options(mmap.MAP_SHARED):
        result = load_checkpoint(path, memory_map=True)
    assert calls[0]["mmap"] is False and calls[0]["weights_only"] is True
    result["weight"].zero_()
    assert original(path, weights_only=True)["weight"].sum() > 0


class UnsupportedPayload:
    pass


@pytest.mark.parametrize("mapped", [False, True])
def test_weights_only_restriction_is_preserved(tmp_path, mapped):
    path = tmp_path / "unsupported.pth"
    torch.save({"payload": UnsupportedPayload()}, path)
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(path, memory_map=mapped)


def test_corrupted_checkpoint_error_is_not_suppressed(tmp_path):
    path = tmp_path / "bad.pth"
    path.write_bytes(b"PK\x03\x04broken")
    with pytest.raises(RuntimeError):
        load_checkpoint(path, memory_map=True)
