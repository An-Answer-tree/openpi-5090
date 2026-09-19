# H7.1 Protocol: Exact-Attention Local Layer Scan

## Question

Is layer 9 a stable optimum within the action expert's middle layers, or an
artifact of the coarse layer 6/9/12 scan?

## Fixed Setup

Reuse H7 without changing the teacher, student, dataset split, noisy actions,
negative control, probe architecture, optimization, or evaluation:

- Teacher: frozen `agentview+wrist` SFT step 29,999.
- Student: frozen backview Flow-only step 4,999.
- New layers: 7, 8, 10, and 11. Together with H7, this covers every layer from
  6 through 12.
- Probe seeds: 11, 29, and 47.
- Optimization: 500 steps, global micro-batch 8, accumulation 4, effective
  batch 32, Adam at `1e-3`.
- Evaluation: 32 fixed episode-held-out batches and the same top-25% teacher-
  advantage hard subset.
- Runtime: two-device FSDP, 16 CPU, and 48 GB host memory on the normal `batch`
  partition; no model checkpoint.

## Metrics And Decision

Report the same correct-minus-shuffled cosine gaps, episode-bootstrap 95%
confidence intervals, explained variance, target variance, and contribution
norms as H7. Apply the unchanged usability thresholds from H7.

After completion, combine these results with the existing layer 6/9/12 results.
Call layer 9 a robust local optimum only if it remains the best usable layer by
at least `0.02` overall gap. If an adjacent layer wins or lies within `0.02`,
treat the middle layers as a plateau and select the simplest representative
using task-success evidence rather than probe gap alone.

This scan measures recoverability. It does not establish causal use by the
student or improved LIBERO success.
