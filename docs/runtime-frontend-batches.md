# Multi-utterance frontend replay: isolated experiment

The adopted frontend graph currently handles B1 only. Existing measurements showed feature/H2D
work falling from0.236 to0.124ms atB1, with small/neutral effects on larger-batch workloads whose
frontend remains eager. `FrontendReplay(max_batch=8)` now prepares an opt-in B2..8 experiment.
The public pipeline retains a private batch limit1; no default expansion or speed claim is made.

The same STFT/mel/normalization at the same input shape is captured. The cache remains bounded
at32 entries per variant. Capture-stream/workspace isolation, cross-stream event ordering and
independent cloned outputs are retained. Tests extend the existing global/utterance MVN, fused /
unfused, cross-stream lifetime, graph-cap and autocast checks to mixedB3/B8 shapes; unsupported
larger batches remain eager. GPU correctness/performance are pending.

`profile_frontend_batches.py` first checks exact real-model features and separately measures
feature/H2D time and graph allocation changes atB3/B8. It then compares B3 timestamp plain/beam,
B8 translation and drafted timestamp ASR, English73/Japanese12 on1B/102M,11 alternating pairs.
B1 graph replay remains enabled in both modes. Each mode owns a separate frontend graph cache;
the warm capture guard now compares each mode to its own warmed count. Full token/time/text /
segment equality and no runtime decoder fallback are required. Call diagnostics are separate.

Graph private pools may add substantial memory and cold capture work. The recorded feature
allocation delta includes a retained output/library workspace; end-to-end peaks include both
resident variants. These are not isolated deployment-memory savings. Existing B1 graph gains
must not be reported as a new B3/B8 gain. No change is adopted unless measured request/throughput
benefits justify the added capture/memory cost.
