# H9 Batch-Scaling Protocol: ACPD-v2 30K

## Question

How do two-device physical global BS32 and four-device physical global BS64
affect layer-10 ACPD-v2 runtime and memory on RTX 5090?

## Training Control

Use the H9 layer-10 exact-contribution target, frozen
`agentview+wrist` teacher at step 29,999, the same backview fixed dataset,
student initialization, seed 42, LoRA ranks, loss weights, and 30K cosine
schedule. Set gradient accumulation to one in both runs.

| Run | GPUs | Global batch | Local batch/GPU | Steps |
|---|---:|---:|---:|---:|
| H9-scale-a | 2 | 32 | 16 | 30,000 |
| H9-scale-b | 4 | 64 | 16 | 30,000 |

The local batch is intentionally matched so per-GPU activation memory is
comparable. H9-scale-b processes twice as many samples per optimizer step, so
the runs do not have equal sample budgets at 30K steps. Keep the schedule and
learning rate unchanged to measure the requested infrastructure configurations;
do not interpret their task-success difference as a controlled batch ablation.

Both jobs must remain finite and avoid OOM. Record peak sampled GPU memory,
step time after warm-up, samples per second, total wall time, and checkpoint
completion. If a job fails before producing a checkpoint, mark that
configuration unsupported and do not infer task success.

## Time Estimate

The four-device BS64 run reached step 100 at about 6.0 seconds per optimizer
step and 31,464 MiB peak sampled memory per card. This gives about 10.7
samples/second and an initial 30K estimate of 50 hours plus checkpoint writes.
The four-device BS32 reference was 8.1--8.6 seconds per step before the BS64
job started on the same node, or about 3.8 samples/second.

H9-scale-a remained finite through step 83 at 12.0--12.6 seconds per step and
31,388 MiB peak sampled memory per card. It was then stopped by operator
decision because the four-device configurations provide lower wall time. It
did not reach the first checkpoint save.

At 30K optimizer steps, BS64 processes 1.92 million samples while BS32
processes 0.96 million. For an equal 0.96-million-sample budget, BS64 would use
15K steps and take about 25 hours before checkpoint overhead.

Detailed measurements and the interpretation boundary are recorded in
`analysis.md`.
