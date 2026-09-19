# H8 Protocol: ACL-only Task Success at 5K Steps

## Question

Does ACL alone explain the H5 task-success gain attributed to Full ACPD?

## Training Control

Train one backview ACL-only student for 5,000 steps:

| Loss | Weight |
|---|---:|
| Flow matching | 1.0 |
| Cue prediction | 0.0 |
| ACL | 0.5 |

Initialization, frozen agentview+wrist teacher, fixed dataset, seed 42, LoRA
ranks, physical global batch 32, two-device FSDP, and learning-rate schedule
match H5. Save only the step-4,999 checkpoint. Do not copy the dataset or
checkpoints.

## Evaluation

Evaluate the checkpoint on the same fixed backview camera, four LIBERO suites,
50 trials per task, seed 7, and replan interval 5 used by H5. Report suite and
pooled success over 2,000 episodes. Compare paired episode outcomes against
both Flow-only and Full ACPD with 10,000 task-stratified bootstrap samples.

## Decision Rule

- If Full ACPD exceeds ACL-only by at least 1.5 percentage points and the paired
  95% confidence interval excludes zero, the old Cue has evidence beyond ACL.
- If ACL-only is within 0.5 points of Full ACPD, treat H5's gain as explained by
  ACL at this screening resolution.
- Otherwise, component attribution is inconclusive.

This is a single-seed 5K screen. It does not include repeated seeds, longer
training, or additional student views.
