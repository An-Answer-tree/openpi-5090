# Backview SFT BS64 Baseline Protocol

## Question

Does layer-10 ACPD-v2 improve 5K task success over standard backview-only SFT
when both use four GPUs and physical global batch size 64?

## Training Control

Train standard pi0.5 LoRA with only the supervised flow-matching loss. The model
receives only `backview_image`; no teacher, privileged view, ACL, or ACPD target
is used.

| Setting | Value |
|---|---|
| Dataset | `libero_multiview_tuned_6view_lerobot` |
| Student initialization | pi0.5 base |
| Student view | backview |
| LoRA | PaliGemma rank 16; action expert rank 32 |
| Seed | 42 |
| Devices | 4 RTX 5090 with FSDP |
| Global batch | 64 physical; 16 samples per GPU |
| Optimizer steps | 5,000 |
| Learning rate | 1K warmup; cosine `2.5e-5` to `2.5e-6` over 30K steps |
| Checkpoint | step 4,999 |

The learning-rate schedule intentionally remains the same 30K schedule used by
H9-scale-b. This isolates the method at its shared 5K checkpoint instead of
retuning the baseline for a short run.

## Evaluation

Evaluate checkpoint 4,999 on the four LIBERO suites with 500 episodes per suite.
The primary metric is pooled success over 2,000 episodes. Compare against the
H9-scale-b checkpoint at step 4,999 using task-stratified paired bootstrap with
10,000 resamples and seed 42.

The ACPD-v2 result is supported if its pooled success exceeds this SFT baseline
by at least 1.5 percentage points and the paired 95% confidence interval lower
bound is greater than zero. Training loss is used only as a health check.

## 30K Continuation

Resume the same run from checkpoint `4999`, whose restored optimizer step is
5,000, and train to 30,000 steps. Keep the seed, physical global batch size,
FSDP topology, optimizer state, and original 30K cosine schedule unchanged.

Reconstruct the shuffled sampler at logical batch 5,000 by advancing only its
random-number state. Do not read the skipped images. Save every 5,000 steps and
retain all checkpoints so the 10K--30K learning curve can be evaluated.
