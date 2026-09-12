# H4 Protocol: ACPD Component Ablation at 2K Steps

## Question

Which ACPD component changes early LoRA student optimization: privileged Cue,
action correlation loss (ACL), or their combination?

## Rationale

The H3 layer comparison only tests whether a second ACPD head is useful. It
cannot compare ACPD with ordinary fine-tuning. A historical LoRA SFT log has a
different RNG split from the distillation trainer, so it is exploratory rather
than a controlled baseline.

The Cue-to-flow loss ratio falls sharply during the first 1K steps and is about
2% by step 4,900. A 2K run is therefore sufficient to measure the period where
the auxiliary objective changes most. It also avoids writing diagnostic
checkpoints that will not be evaluated.

## Controlled Runs

Use the backview student and layers `{6, 12}` for all runs:

| Run | Flow | Cue weight | ACL weight |
|---|---:|---:|---:|
| Flow-only | 1.0 | 0.0 | 0.0 |
| Cue-only | 1.0 | 0.2 | 0.0 |
| ACL-only | 1.0 | 0.0 | 0.5 |
| Full ACPD | 1.0 | 0.2 | 0.5 |

The Full ACPD curve is the first 2K steps of completed job 127143. Its dataset
was a byte copy of the same fixed dataset; all new runs read the original
shared path without copying. Model initialization, seed 42, shuffled sample
order, teacher, batch size 32, two-device FSDP, LoRA ranks, 1K warmup, and 30K
cosine schedule remain fixed. No run uses the historical teacher-reliability
gate, which is absent from the manuscript objective.

All three new runs retain the complete teacher and auxiliary forward graph even
when a loss weight is zero. This keeps RNG use and logged diagnostics matched.
They run for 2,000 optimizer steps, log every 100 steps, and do not save a final
checkpoint.

## Metrics

Primary metric: mean `supervised_loss` over logged steps 100-1,900. Secondary
metric: mean over steps 1,000-1,900. Lower is better.

Also report student and teacher task-dimension flow MSE, teacher-better ratio,
component losses, gradient norms, wall time, and peak GPU memory. Training loss
is an optimization proxy, not a policy-quality measurement.

## Decision Rule

A component changes early convergence only if its primary and final-window
supervised losses move in the same direction by at least 1% relative to the
Flow-only run. If the change is smaller, treat convergence as indistinguishable
at this resolution. If Cue and ACL have opposing effects, tune or schedule only
the harmful component. If none changes supervised loss, select the next test by
task success rather than sweeping loss weights.

Teacher diagnostics are interpreted separately: a low teacher-better ratio
would motivate reliability gating; a consistently strong teacher would instead
shift attention to cue shortcuts, LoRA capacity, and task-level evaluation.

This is a single-seed screening experiment. Any claimed policy improvement
requires checkpoint evaluation and replication.
