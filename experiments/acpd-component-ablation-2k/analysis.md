# H4 Analysis: ACPD Component Ablation at 2K Steps

## Result

H4 is not supported at the pre-registered 1% resolution. Relative to the
trainer-matched Flow-only run, neither Cue, ACL, nor their combination changed
both supervised-loss windows by at least 1%.

| Run | Mean loss, steps 100-1,900 | Delta vs. Flow | Mean loss, steps 1,000-1,900 | Delta vs. Flow | Time to 2K | Peak MiB/GPU |
|---|---:|---:|---:|---:|---:|---:|
| Flow-only | 0.050495 | 0.00% | 0.042920 | 0.00% | 1:48:53 | 31,390 |
| Cue-only | 0.050900 | +0.80% | 0.043340 | +0.98% | 2:08:18 | 31,389 |
| ACL-only | 0.050337 | -0.31% | 0.042650 | -0.63% | 1:41:16 | 31,378 |
| Full ACPD | 0.050553 | +0.11% | 0.042990 | +0.16% | 1:48:18 | 31,389 |

Positive deltas are worse. Metrics use the 19 logged points in the primary
window and 10 points in the final window. The Full ACPD row is the matching
first-2K prefix of job 127143. Wall times are operational measurements and are
not interpreted as method effects because the jobs ran on different nodes and
under different shared-storage loads.

## Sanity Checks

- All runs completed 2,000 finite optimizer steps with physical global batch
  32 and two-device FSDP.
- Flow-only, Cue-only, and ACL-only used identical teacher task losses in the
  primary window: 0.015953. The teacher was better than the student on about
  99.9% of samples, so weak teacher targets do not explain the result.
- Cue-only retained nonzero selector and predictor gradients. At step 1,900,
  cue cosine reached 0.996 and normalized cue feature standard deviation
  reached 0.975. The cue objective was optimized without variance collapse.
- ACL-only reduced the final-window action-correlation loss from 0.06640 to
  0.06555 and increased the mean LoRA gradient norm from 0.0352 to 0.0643.
  ACL therefore changed the optimization signal even though supervised flow
  convergence remained effectively unchanged.

## Interpretation

Cue-only was slightly worse than Flow-only at 16 of 19 logged primary-window
steps, while ACL-only was slightly better at 14 of 19. Both effects remain
below the locked threshold. Full ACPD was almost identical to Flow-only,
consistent with the two weak trends cancelling, but this is a mechanistic clue
rather than evidence of interaction.

The cue loss becoming easy does not prove that the selector transfers
task-relevant privileged information. Joint selector-predictor adaptation can
produce a diverse, predictable cue without improving the student's action
fit. Conversely, unchanged supervised loss does not rule out a policy-quality
effect from representation regularization or action-shape alignment.

## Decision

Do not tune Cue or ACL weights from this proxy. The next discriminating test is
a checkpoint-matched backview task-success comparison between Flow-only and
Full ACPD. Run the component-level task evaluations only if Full ACPD first
shows a useful success-rate gain. Any task-success claim remains exploratory
until it is replicated across seeds.

