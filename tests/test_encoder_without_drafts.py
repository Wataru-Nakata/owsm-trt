"""The experimental export preserves cross KV and never calls the external draft head."""

import pytest
import torch
from profile_encoder_variants import CrossOnlyGraph
from test_model import TINY
from torch import nn

from owsm_trt.build import _EncoderGraph
from owsm_trt.modeling import OWSMModel


class ForbiddenHead(nn.Module):
    def forward(self, _):
        raise AssertionError("unused CTC head was evaluated")


@pytest.mark.parametrize("batch,frames", [(1, TINY.num_frames), (3, TINY.num_frames), (2, 85)])
@torch.inference_mode()
def test_cross_only_graph_keeps_kv_and_binding_shape(batch, frames):
    torch.manual_seed(61)
    network = OWSMModel(TINY).eval().requires_grad_(False)
    features = torch.randn(batch, frames, TINY.n_mels)
    expected = _EncoderGraph(network)(features)
    network.ctc = ForbiddenHead()
    actual = CrossOnlyGraph(network)(features)
    for left, right in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert actual[2].dtype == torch.int32
    assert actual[2].shape == expected[2].shape
    assert not actual[2].any()
