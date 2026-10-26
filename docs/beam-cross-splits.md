# Cross-attention slice and query layout sweep

Reference:4d8466a (the adopted KV version). This is a separate opt-in N experiment: changing the
cross-attention key slices changes fp32 softmax reduction order, so exact output compatibility is
measured, not assumed. Default runtime behavior is unchanged before validation.

The previous grouped-query experiment kept NS=1 at R=5. That reduced cross-attention units from
80 to16 on the1B model. The new sweep compares NS=2/4/8 both with independent queries and grouped
sibling queries. Grouping shares K/V loads; key splitting restores more independent units. The
persistent grid still uses132 CTAs on H200 and the existing row/token routing limits. All sequence
positions, search rules, weights and dtypes remain unchanged.

The prepared allocation runs existing correctness tests, grouped/shared attention checks and six
new split tests. The split tests cover B2K3, repeated execution and empty slices with65 encoder
keys, allowing small numerical differences versus the original reduction but requiring finite,
deterministic outputs. Real English73/Japanese12 comparisons record all token/segment differences,
WER/CER, latencies and actual dispatch. Six configurations execute serially, each on the same loaded
checkpoint/engines as its alternating reference. No new speed or quality claim is made before those
results exist. One GPU, same approved group/queue/resource, <=192 account GPUs, local staging/logs.
