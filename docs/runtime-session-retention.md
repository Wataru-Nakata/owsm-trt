# Session retention experiment (not GPU measured)

Candidate22 also covers session eviction. The production encoder-decoder factory retains two
sessions, keyed by utterance batch and beam width. A B1 beam request followed by a five-input B3
request uses `(1,5)`, `(3,5)` and `(2,5)`. Repeating this sequence recreates all three sessions under
the two-entry LRU. Retaining three sessions is an E-intent hypothesis; its speed and GPU output
compatibility have not yet been measured. The public API and default retention count remain unchanged.

`scripts/profile_session_retention.py` compares the existing two-entry policy with a private
three-entry control. It selects12 inputs each from English and Japanese in the expanded corpus,
uses two distinct six-input cycles, and alternates the policies over seven repeats. Each cycle is
one single-input request followed by one five-input request at B3, with beam5 and timestamps.
Every full output must match across policies and repeated calls. This is a mixed-shape workload;
reference timings intentionally include session setup and graph capture caused by eviction.
They must not be presented as a decoder kernel improvement or a gain for an ordinary fixed-shape
warm request.

Model loading, audio reads and initial warmup are separate. A diagnostic pass counts session
creations without keeping evicted sessions alive: six creations over two cycles are expected
for the reference and none for the candidate. Timed calls do not install this observer. Candidate
captures must stay constant after warmup; output changes or decoder fallback invalidate the run.
Both session pools remain resident, so reported allocator peaks describe the combined experiment,
not standalone deployment memory. A separate memory assessment is required before adoption.

Three small CPU fixtures cover the default two-entry policy, three-shape reuse and LRU refresh on
a cache hit. They allocate no model tensors and do not establish CUDA graph lifetime safety.
The CTC-only session cache uses a separate policy and is unchanged.

Run only inside an allocated compute job, with the environment, model, engines and corpus staged
on node-local storage:

```bash
python scripts/profile_session_retention.py --model MODEL --data LOCAL_CORPUS \
  --languages eng jpn --repeat 7 --out LOCAL_RESULTS
```

The private runtime field is only an experiment control. No new performance number or production
retention option has been adopted.
