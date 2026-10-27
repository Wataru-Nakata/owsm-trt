"""Audio decoding/preprocessing checks run only in an allocated compute job."""

import os

import numpy as np
import pytest
import torch
from profile_audio_io import legacy_load_audio

from owsm_trt.audio import load_audio

pytestmark = pytest.mark.skipif(not os.environ.get("PBS_JOBID"), reason="audio work requires compute allocation")


@pytest.mark.parametrize("rate", [16000, 44100])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("subtype", ["PCM_16", "FLOAT"])
def test_mono_view_preserves_decode_resample_and_tensor_lifetime(tmp_path, rate, channels, subtype):
    import soundfile as sf

    rng = np.random.default_rng(42)
    data = rng.uniform(-0.5, 0.5, (rate // 10, channels)).astype(np.float32)
    data[:4] = np.array([0.0, -0.0, 0.25, -0.25], dtype=np.float32)[:, None]
    path = tmp_path / "audio.wav"
    sf.write(path, data, rate, subtype=subtype)
    expected = legacy_load_audio(path)
    actual = load_audio(path)
    assert actual.dtype == torch.float32 and actual.device.type == "cpu" and actual.ndim == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # A later file read cannot overwrite the array storage owned by a retained mono tensor.
    load_audio(path).fill_(9)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nonfile_inputs_keep_existing_conversion_and_errors():
    data = torch.tensor([[0.0, -0.0, 0.25, -0.25], [0.5, 0.1, -0.1, -0.5]])
    torch.testing.assert_close(load_audio(data), data.mean(dim=0), rtol=0, atol=0)
    with pytest.raises(ValueError, match="expected 1-D or 2-D"):
        load_audio(torch.zeros(2, 2, 2))
