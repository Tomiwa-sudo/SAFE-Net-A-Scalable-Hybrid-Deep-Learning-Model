# Why the architecture changed: LSTM/sequence windowing removed

## The problem, in one sentence
CICIoT2023 rows carry no timestamp — each row is already an aggregated
summary statistic over a packet window computed by the dataset's own
feature extractor — and inspection of the actual `Merged*.csv` files
showed 15–22 distinct attack categories interleaved within any 200-row
span. There is no genuine temporal or session-level order left to model,
so an LSTM-based "sequence" window built from consecutive rows was never
capturing real sequential behavior — first because the original pipeline
shuffled the full dataset before windowing (a straightforward bug), and
second, more fundamentally, because even the correctly-ordered rows in
the source files aren't temporally meaningful at the granularity needed.

## What this means for every LSTM/"sequence" claim in the original draft
- "LSTM Autoencoder for unsupervised **temporal** representation learning" → false as stated.
- "captures **sequence-level** behaviour" → false as stated.
- "well suited for detecting changes that occur **over time**" → false as stated.
- The reported "0.013 ms per packet" latency number was computed by
  dividing a *batched* forward-pass time by `batch_size * seq_len`,
  compounding the problem: it wasn't measuring per-packet latency in any
  real sense even under the old architecture.

## The fix
Replace the `LSTM Autoencoder + windowed classifier` with a **Dense
(fully-connected) Autoencoder + fusion MLP classifier operating on one
flow row at a time**. This is a strictly honest re-framing, not a
downgrade of the paper's actual idea:

- The genuinely novel/useful part of the original contribution —
  **fusing an unsupervised reconstruction-based latent representation
  with the raw features, feeding both into a supervised classifier** —
  survives completely intact. Nothing about that idea depended on
  windowing or LSTMs.
- What's removed is only the incorrect temporal/sequential framing
  layered on top of it.
- As a side benefit, per-packet latency is now trivially honest to
  measure: one row goes in, one prediction comes out, with no `seq_len`
  or batch-size division tricks required.

## Recommended rewording (search-and-replace + two paragraph rewrites)
| Original | Replace with |
|---|---|
| SAFE-Net (**S**equence-Autoencoder Fusion Embedding Network) | SAFE-Net (**S**calable Autoencoder Fusion Embedding Network) |
| "LSTM-based Sequence Autoencoder" | "Autoencoder" |
| "unsupervised temporal representation learning" | "unsupervised representation learning" |
| "captures sequence-level behaviour" | "captures reconstruction-based anomaly signal" |
| Algorithm 1: "Hybrid Sequence Autoencoder IDS" | "Hybrid Autoencoder IDS" |
| Algorithm 1, step 2 ("Sequence Generation: Construct sliding windows...") | removed entirely |
| Table I row "LSTM Autoencoder (AE)" | "Autoencoder (AE)"; remove any `seq_len`/window rows |
| Training Setup bullet "Temporal Modeling: The LSTM Autoencoder processes sequential data..." | replace with an "Unsupervised Signal" bullet about reconstruction-error features augmenting the supervised classifier |

Section III.B ("Data Processing") needs a real rewrite, not a word swap
— it currently explains LSTM-AE sequence-modeling machinery that no
longer applies. Replace with a short paragraph on why a dense
autoencoder's reconstruction error is a useful unsupervised signal for
anomaly-adjacent detection, fused with raw features for the classifier.

## For the reviewer response letter
Both reviews are compatible with this change and neither needs a
defensive response — if anything, this preempts a criticism neither
reviewer explicitly made but easily could have. Suggested language:

> "In revising the manuscript, we identified that the CICIoT2023 feature
> set does not retain packet-level timestamps, and that flow records from
> different attack scenarios are interleaved within our merged CSV files.
> We concluded that windowed LSTM sequence modeling was not an accurate
> characterization of the signal being learned, and we have replaced the
> LSTM Autoencoder with a Dense Autoencoder operating on individual flow
> records, fused with raw features for classification. This preserves the
> core hybrid supervised/unsupervised contribution while removing an
> inaccurate temporal-modeling claim. We have also corrected the latency
> measurement methodology accordingly (Section IV.B) and added the
> baseline, ablation, generalization, and multiclass analyses requested
> below."
