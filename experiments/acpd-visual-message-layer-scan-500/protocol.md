# H6.1 Protocol: Visual-Message Layer Scan at 500 Steps

## Question

Does action-expert depth change how well a frozen backview student predicts the
current action-conditioned teacher visual-message proxy?

## Hypothesis

Layer 9 should balance visual context and action semantics better than layers 6
and 12, producing the largest correct-minus-shuffled cosine gap.

## Experiment

Compare three otherwise matched probes with `align_layer` set to 6, 9, or 12.
The existing layer-6 job is the first run; submit new layer-9 and layer-12 runs.

- Teacher: frozen `agentview+wrist` step-29,999 checkpoint.
- Student: frozen pi0.5 base model with backview input.
- Data: `libero_multiview_tuned_6view_lerobot`.
- Optimization: one 1024-by-1024 linear probe, Adam at `1e-3`, 500 steps.
- Runtime: seed 42, physical global batch 32, two-device FSDP.
- Output: metrics only; no model checkpoint.

The new jobs use `num_workers=0` because the layer-6 job stalled before its
first batch with multiprocessing workers waiting on shared storage. This is a
runtime-only change; the sampler seed and scientific inputs remain fixed.

## Metrics and Decision

For each layer, average the logged step 400, 450, and 499 values as the final
100-step estimate. Report:

- correct cosine;
- batch-shuffled cosine;
- their difference;
- target standard deviation;
- probe gradient norm.

A layer passes the recoverability screen when its cosine gap is at least 0.10,
the target variance is nonzero, and gradients are finite. Select a preferred
layer only if its gap exceeds the runner-up by at least 0.02; otherwise treat
the passing layers as indistinguishable.

## Scope

This compares the existing heuristic proxy, not the transformer's true Q/K/V
visual-attention contribution. The visual and action features are not in a
learned common projection space, so the result can only measure sensitivity of
this probe to action-layer depth. It cannot establish the best ACPD layer or
policy improvement.
