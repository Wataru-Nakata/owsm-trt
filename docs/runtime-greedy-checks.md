# Plain greedy completion check intervals

After the measured plain mask/argmax/state fusion, the host still checks completion every4 steps
in GreedySession. This can execute up to3 decoder calls after a post-prefill EOS, or4 when EOS is
selected directly from prefill. A more frequent check also costs synchronization; its net effect
must be measured. Beam already has its separately measured B1/B3 policy.

`scripts/profile_greedy_checks.py` keeps the accepted public optimizations, checkpoint, engines,
precision and decoder dispatch fixed and compares intervals4/2/1. Session.chunk is changed only
inside this benchmark; the production greedy default stays4. All73 English inputs and12 Japanese
inputs are used with plain greedy, timestamped plain greedy and translation;102M uses English.
Warm synchronized wall samples alternate variant order over11 repeats. First-use/capture costs and
audio loading are recorded separately, and every returned token/segment must match interval4.

A separate diagnostic pass records actual decoder calls, host checks and the first EOS selection
index per row. Selection0 is prefill, whose decoder call is excluded from the search-call count.
The batch's excess is actual decoder calls minus its last row's first EOS index; a length-limited
request with no EOS has no excess claim. GPU state is copied only after each diagnostic request,
never added to the timed per-token loop. Artificial CPU checks cover prefill EOS, later EOS and
length-limit accounting. GPU speed and output agreement are unmeasured until the job completes.
