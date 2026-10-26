# Wider-row Gluon experiment

R=10/15 (B=2/3, K5) currently dispatches to TensorRT. Prior R16 Gluon measurements in the
source were slower than TensorRT; changing the limit alone is not assumed to help. Recent
shared cross-KV and grouped-query work was measured only at R5. This isolated N-class
experiment checks whether their memory reuse makes R10/15 competitive.

`profile_beam.py --gluon-max-rows 16` creates a separate candidate backend; ordinary routing
remains R<=8 and <=16 tokens. Existing Gluon tiles support up to32 tokens; this experiment
uses10/15, padded MT16, HL2. GEMM units/persistent CTA count, resident-grid requirements and
self-KV layout do not change. Cross-attention loops cover more than one head unit per CTA
when needed. Tests cover distinct utterance KV, offset0 and offsets through70 in capacity128,
fp32 decoder logits/new-KV, exact shared/expanded results, graph replay and changed inputs.

The allocation first runs expanded GPU tests and real-checkpoint fp32 checks at R10/15.
It compares three isolated candidate wrappers with the current TensorRT reference: shared
independent queries at automatic splits, grouped queries with four slices, and eight slices.
English73/Japanese12, B2/B3, seven alternating repetitions, K5/timestamps; all actual dispatch,
full token/time/segment output, WER/CER, diagnostics, cold setup and uninstrumented wall
measurements are saved. Different floating-point arithmetic is N-class and may change outputs.
There are no reference timestamp annotations; differences are agreement, not correctness errors.

The same allocation independently repeats selective checkpoint mapping (three pairs/model)
and includes an unmodified production-settings worker for1B/102M, recording actual torch.load
flags. Startup validation and decoder performance are separate. No runtime routing limit is
changed by the experiment. Speed and wider-row parity are unmeasured until results exist.

Reproduce inside an allocation with `scripts/profile_gluon_rows.pbs`; prepared code/engines/audio
are staged locally and only coarse progress/final artifacts are published. Submission uses the
standing serial1 GPU authorization and a fresh account-capacity/queue/duplicate check.

## First validation attempt:2773118

Source80fd2e8 stopped after14passing tests at shared/expanded graph-replay comparison B2/K3
(capacity128): one row's logits differed. No load/beam benchmark ran. Inspection found that the
test retained graph/input/output tensors but not both closures owning externally allocated
workspaces. CUDA graph replay requires those addresses to remain valid. The test now retains
both owners; all exact comparisons remain. This is a suspected cause pending GPU revalidation,
not an assertion that the kernel is correct. Raw failure: `outputs/runtime-2773118/tests.log`.
