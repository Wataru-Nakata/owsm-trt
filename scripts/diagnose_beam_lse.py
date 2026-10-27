"""Save per-step N-class branch differences; no inference/timing claim from synthetic logits."""

import argparse
import json
import os
from pathlib import Path

import torch
from test_beam import session

from owsm_trt import beam_kernels
from owsm_trt.runtime import TimestampRules


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('diagnostic requires compute allocation')
    records, finals = {}, {}
    for fused in (False, True):
        device = torch.device('cuda')
        s = session(batch=1, beam=5, device=device, fused_lse=fused, use_cuda_graph=False,
                    rules=TimestampRules(7, 5, 6, 0, 8, device))
        original = s._topk
        records[str(fused)] = []

        def observed(prefill_logits=None):
            top, indices = original(prefill_logits)
            selected = s._candidates.clone()
            old_error = s.numerical_error.clone()
            logits = s.logits[:, -1]
            torch_candidates = beam_kernels.candidates(s, torch.log_softmax(logits, -1)).clone()
            torch.testing.assert_close(torch.isfinite(selected), torch.isfinite(torch_candidates), rtol=0, atol=0)
            torch.testing.assert_close(old_error, s.numerical_error, rtol=0, atol=0)
            finite = torch.isfinite(selected)
            delta = float((selected[finite] - torch_candidates[finite]).abs().max()) if finite.any() else 0.
            assert delta <= 2e-5
            want_scores, want_indices = torch_candidates.view(1, -1).topk(5)
            actual_in_reference = torch_candidates.view(1, -1).gather(1, indices).sort(descending=True).values
            valid = torch.isfinite(want_scores)
            gap = float((want_scores[valid] - actual_in_reference[valid]).max()) if valid.any() else 0.
            assert gap <= 2 * delta + 1e-7, (gap, delta)  # Selected alternatives must be within measured rounding error.
            records[str(fused)].append(dict(step=int(s.step), max_candidate_delta=delta,
                selection_gap_in_torch_scores=gap, selected_indices=indices.tolist(),
                torch_indices_same_input=want_indices.tolist(), selected_scores=top.tolist(),
                torch_scores_same_input=want_scores.tolist(), parent_scores=s.score.tolist(),
                parent_prefixes=s.seq[:, :int(s.pos)].tolist()))
            s._candidates.copy_(selected)
            return top, indices

        s._topk = observed
        output = s.decode(torch.tensor([[2, 2, 7]], device=s.device), 13)
        finals[str(fused)] = dict(output=output, seq=s.seq.tolist(), score=s.score.tolist(),
            best_score=s.best_score.tolist(), best_seq=s.best_seq.tolist(), best_end=s.best_end.tolist())
    a, b = finals['False'], finals['True']
    report = dict(steps=records, final=finals, output_equal=a['output']==b['output'],
        side_beams_equal=a['seq']==b['seq'], same_input_masks_errors_equal=True,
        selection_differences_bounded_by_measured_rounding=True,
        scope='Synthetic constant logits: independent torch-vs-blockwise branches and same-input candidate scores. '
              'Not evidence of model quality or speed; full model comparison remains required.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print({k:v for k,v in report.items() if k not in ('steps','final')})


if __name__ == '__main__':
    main()
