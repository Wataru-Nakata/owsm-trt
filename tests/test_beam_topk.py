"""Selection-only tensor tests; exact ranking without ties, explicit tie-set semantics otherwise."""

import pytest
import torch

from owsm_trt.runtime import beam_topk


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("beam", [2, 5, 8])
@pytest.mark.parametrize("vocab", [3, 17])
@pytest.mark.parametrize("pattern", ["distinct", "near_tie", "sparse", "all_invalid", "tie"])
def test_local_topk_scores_and_original_parent_token_indices(batch, beam, vocab, pattern):
    g = torch.Generator().manual_seed(123)
    width = beam * vocab
    scores = torch.stack([-torch.randperm(width, generator=g).float() / 8 for _ in range(batch)])
    if pattern == "near_tie":
        # Distinct representable neighbors: epsilon must never be added to choose a tie winner.
        scores *= 8 * torch.finfo(torch.float32).eps
        scores -= 1
        assert all(row.unique().numel() == width for row in scores)
    elif pattern == "sparse":
        scores.fill_(-torch.inf)
        scores[:, -1] = -3
    elif pattern == "all_invalid":
        scores.fill_(-torch.inf)
    elif pattern == "tie":
        scores.zero_()
    expected, expected_indices = beam_topk(scores, beam)
    actual, indices = beam_topk(scores, beam, local=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(scores.gather(1, indices), actual, rtol=0, atol=0)
    assert all(len(set(row)) == beam for row in indices.tolist())
    if pattern in ("distinct", "near_tie"):
        torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph check requires a compute allocation")
@pytest.mark.parametrize("batch", [1, 3])
@torch.inference_mode()
def test_local_topk_large_vocabulary_graph_reuse(batch):
    beam, vocab = 5, 50002
    width = beam * vocab
    scores = torch.stack([-torch.randperm(width, device="cuda").float() / 8192 for _ in range(batch)])
    # Live rows, timestamp-sized intervals, <K valid extensions and all-invalid groups.
    for offset in (0, 1, 2):
        scores[:, offset::11] = -torch.inf
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            beam_topk(scores, beam, local=True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, indices = beam_topk(scores, beam, local=True)
    for iteration in range(3):
        if iteration == 1:
            scores.fill_(-torch.inf)
            scores[:, vocab-1] = -5
        elif iteration == 2:
            scores.fill_(-torch.inf)
        graph.replay()
        expected, _ = beam_topk(scores, beam)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(scores.gather(1, indices), actual, rtol=0, atol=0)
