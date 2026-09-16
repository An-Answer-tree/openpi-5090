# H9.1 Protocol: Layer-10 ACPD-v2 5K Screen

## Question

Does the best recoverable layer from H7.1 produce a stronger deployed ACPD-v2
policy than the already running layer-9 H9 policy?

## Training Control

Train one backview policy for 5,000 optimizer steps. Match H9 in initialization,
teacher, dataset, seed 42, LoRA ranks, loss weights, four-device FSDP, global
micro-batch 8, four-step accumulation, and effective batch 32. Keep the H9
schedule of 1K warmup followed by cosine decay from `2.5e-5` to `2.5e-6` over
30K steps so the first 5K updates are directly comparable.

The only scientific change from H9 is the exact-contribution layer:

| Run | Exact-contribution layer | Optimizer steps |
|---|---:|---:|
| H9 | 9 | 5,000 |
| H9.1 | 10 | 5,000 |

Save checkpoint 4,999. Do not copy the dataset or checkpoints.

## Evaluation

Evaluate H9.1 on the same fixed backview camera, four LIBERO suites, 50 trials
per task, seed 7, and replan interval 5. Compare all 2,000 paired episode
outcomes with H8 ACL-only and H9 layer 9 using 10,000 task-stratified bootstrap
samples.

## Decision

- The deployed layer-10 mechanism passes against ACL-only only if pooled success
  improves by at least 1.5 percentage points and the paired 95% confidence
  interval excludes zero.
- Layer 10 is better than layer 9 only if pooled success improves by at least
  1.0 percentage point and the paired 95% confidence interval excludes zero.
- Treat an absolute layer-10 versus layer-9 difference below 0.5 points as
  equivalent at this screening resolution. Otherwise, treat the comparison as
  inconclusive.

This is a single-seed 5K screen. Recoverability metrics motivated the layer
choice but are not policy-success evidence.
