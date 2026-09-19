# H3 Protocol: ACPD Layer Ablation at 5K Steps

## Question

Does applying ACPD at both action-expert layers 6 and 12 improve early backview
training convergence enough to justify two auxiliary heads?

## Hypothesis

Layers 6 and 12 provide complementary intermediate and later action features.
The `{6, 12}` student should therefore converge faster than either `{6}` or
`{12}` alone.

## Controlled Runs

Run three backview students with `align_layers` set to `{6}`, `{12}`, and
`{6, 12}`. All other settings are fixed:

- base pi0.5 initialization and random seed 42;
- frozen agentview+wrist teacher checkpoint at step 29,999;
- fixed multiview dataset and synchronized sample transforms;
- LoRA ranks 16 for PaliGemma and 32 for the action expert;
- two-device FSDP, physical global batch 32, no gradient accumulation;
- 5,000 optimizer steps and logs every 100 steps;
- 1,000-step warmup to 2.5e-5 followed by the original 30K cosine schedule;
- `lambda_cue=0.2`, `lambda_var=0.1`, and `lambda_ACL=0.5`.

Keeping the 30K learning-rate schedule means this experiment measures the
first 5K steps of the intended full run instead of a compressed training run.

## Metrics

The primary metric is the mean logged `supervised_loss` over steps 100-4,900.
This is the discrete convergence AUC normalized by the number of observations.
The secondary metric is mean `supervised_loss` over steps 4,000-4,900.
Lower is better for both.

Also report total loss, action-correlation loss, cue prediction and variance
losses, gradient norms, wall time, and sampled peak GPU memory. Total loss is
secondary because its auxiliary target changes with the selected layers.

## Decision Rule

The two-layer design is justified for early convergence only if `{6, 12}` has
the lowest primary metric and its final-window supervised loss is at least 1%
lower than the better single-layer run. If the final-window difference is less
than 1%, prefer the simpler single layer. If either single layer is better on
the primary and final-window metrics, two-layer ACPD is not necessary for early
loss convergence.

This single-seed 5K experiment cannot establish final policy quality. A
benchmark evaluation or full-length replicated experiment is required before
making a task-success claim.

## Pre-Run Infrastructure Amendment

The scheduler estimated a one-week wait for four contiguous GPUs. Before any
layer run started, job 127114 was submitted as a two-step feasibility check for
two-device FSDP with the same physical global batch of 32. If it completes
without OOM, all three controlled runs will use two-device FSDP, 16 CPUs, and
64 GB host memory. If it fails, the original four-device allocation remains the
fallback. FSDP device count is an infrastructure variable shared by all three
runs and does not change the scientific comparison.

## Execution Amendment

The smoke check completed on an RTX 5090 with two-device FSDP, physical global
batch 32, and a peak of about 31.4 GiB per GPU. The controlled runs therefore
use that two-device configuration. The batch queue estimated a multi-day wait,
so the same scripts were submitted to debug01, which has RTX 5090 GPUs. The
first multi-worker launch stalled while fetching its first batch over NFS. The
complete 123 GiB dataset was copied to debug01 local NVMe for the completed
layers 6+12 and layer 12 runs. The layer 6 run uses the original shared dataset
path on the ordinary batch partition, as requested by the operator. All runs
use `num_workers=8`; the source dataset and model, loss, optimizer, seed, and
layer settings remain unchanged.
