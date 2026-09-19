# H5 Protocol: ACPD Task Success at 5K Steps

## Question

Does Full ACPD improve backview LIBERO task success over trainer-matched
Flow-only training when their early supervised-loss convergence is
indistinguishable?

## Hypothesis

The auxiliary objectives can improve the deployed policy through
representation regularization and action-shape alignment without materially
changing supervised flow loss. Full ACPD should therefore exceed Flow-only by
at least 2 percentage points in pooled four-suite success at 5K steps.

## Training Control

Compare two backview students:

| Run | Flow | Cue | ACL | Checkpoint |
|---|---:|---:|---:|---|
| Flow-only | 1.0 | 0.0 | 0.0 | New matched 5K run |
| Full ACPD | 1.0 | 0.2 | 0.5 | Completed job 127143, step 4,999 |

Both runs use the same pi0.5 initialization, frozen agentview+wrist teacher,
fixed multiview data, seed 42, layers `{6, 12}`, LoRA ranks, physical global
batch 32, two-device FSDP, and 30K learning-rate schedule. The Flow-only run
retains the complete teacher and auxiliary forward graph with zero auxiliary
weights. The completed Full run read a byte-identical local copy of the data;
the Flow-only run reads the original shared path without copying.

## Evaluation

Evaluate each step-4,999 checkpoint using the fixed backview camera profile on
`libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`. Use 50 trials
per task, seed 7, replan interval 5, and the same evaluator for both policies.
Each suite contains 500 episodes, so the primary pooled rate covers 2,000
episodes and equals the unweighted mean of the four suite rates.

The two policies use the same task order and initial states. Report the paired
success difference for every suite and for the pooled 40 tasks. Estimate a 95%
confidence interval with 10,000 paired bootstrap samples, resampling the 50
paired episodes independently within each task. This interval measures rollout
initial-state uncertainty for the fixed benchmark tasks; it does not replace
training-seed replication.

## Decision Rule

- Promote ACPD component evaluations if Full ACPD improves pooled success by
  at least 2 percentage points and the paired 95% confidence interval excludes
  zero.
- Treat an absolute difference below 2 points as indistinguishable for this
  single-seed screen.
- Treat a decrease of at least 2 points as evidence that the current ACPD
  configuration is harmful at 5K steps.

This is an exploratory policy-quality screen, not a final statistical claim.
A positive result requires seed replication and a full-length comparison. A
neutral 5K result does not by itself rule out a later 30K effect.
