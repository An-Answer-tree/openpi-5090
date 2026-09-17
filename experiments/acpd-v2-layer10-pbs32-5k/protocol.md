# H9 Protocol: Layer-10 ACPD-v2 Physical-BS32 Screen

## Question

Does the best recoverable exact-contribution layer improve 5K task success over
ACL-only when ACPD-v2 uses a physical global batch of 32?

## Training Control

Train one backview policy for 5,000 optimizer steps with the following fixed
configuration:

- frozen `agentview+wrist` teacher at step 29,999;
- backview pi0.5 LoRA student initialized from the base checkpoint;
- layer-10 exact attention contribution and the H8 ACL weight;
- four-device FSDP, physical global batch 32, no gradient accumulation;
- seed 42 and the 30K schedule prefix: 1K warmup followed by cosine decay from
  `2.5e-5` to `2.5e-6`.

The effective batch remains 32. The physical batch removes four repeated
teacher/student micro-steps per optimizer update and computes all batch-level
statistics over the complete batch. The first two steps are the runtime gate:
training must remain finite, avoid OOM, and produce nonzero LoRA and
exact-contribution predictor gradients. Save checkpoint 4,999. Do not copy the
dataset or checkpoints.

The stopped micro-batch-8 layer-9 and layer-10 runs produced no evaluation
checkpoint and are not experimental evidence. This experiment tests only the
layer selected by the completed H7.1 recoverability scan.

## Evaluation

Evaluate checkpoint 4,999 with the fixed backview camera on all four LIBERO
suites, 50 trials per task, seed 7, and replan interval 5. Compare the 2,000
paired episode outcomes with the H8 ACL-only checkpoint using 10,000
task-stratified bootstrap samples.

## Decision

H9 passes only if pooled success exceeds H8 ACL-only by at least 1.5
percentage points and the paired 95% confidence interval excludes zero. This is
a single-seed 5K screen; a positive result requires repeated-seed confirmation.
