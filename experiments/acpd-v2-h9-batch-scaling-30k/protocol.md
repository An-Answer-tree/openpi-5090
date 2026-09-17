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

The current four-device BS32 H9 run is about 8--10 seconds per optimizer step
after compilation. Both scaling runs use local BS16, so the initial estimate is
12--16 seconds per step:

- H9-scale-a: about 100--135 hours (4.2--5.6 days), plus checkpoint overhead;
- H9-scale-b: about 100--145 hours (4.2--6.0 days), with more FSDP communication
  but twice the samples per step.

These are estimates. Measurements after the first 100 steps replace them.
