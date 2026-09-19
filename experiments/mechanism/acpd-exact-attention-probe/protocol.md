# H7 Protocol: Exact Privileged-View Attention Probe

## Question

Can an adapted backview student predict the information that a privileged
`agentview+wrist` teacher actually reads from its two image streams?

## Motivation

The original ACPD target is jointly learned and can be reconstructed through
teacher action tokens that share the student's noisy action and flow time. The
discarded H6 proxy did not fix this: it compared hidden states and raw visual
tokens outside the transformer's learned Q/K/V spaces, and its normalized dot
products were scaled close to zero. H7 measures the real attention contribution
instead of defining another learned cue.

## Frozen Setup

- Teacher: `/opt/liutong/openpi_checkpoints/fixed_dataset/sft/agentview_wrist/29999/params`.
- Student: trainer-matched Flow-only backview step 4,999 checkpoint.
- Data: `libero_multiview_tuned_6view_lerobot`; no data or checkpoint copies.
- Layers: action-expert layers 6, 9, and 12 in one forward pass on identical
  examples and identical noisy actions.
- Probe: one linear map per layer and teacher view, trained with normalized MSE.
  The teacher and student remain frozen.
- Optimization: three probe initialization seeds, 500 optimizer steps, physical
  batch 8, four-step accumulation, effective batch 32, Adam at `1e-3`.
- Runtime: two-device FSDP, `num_workers=0`, seed 42, no model checkpoint.

## Episode Split

Shuffle episode IDs once with NumPy seed 42. Reserve the final 10% of episodes,
rounded up, for validation. Train probes only on the remaining episodes. Report
metrics over 32 deterministic validation batches; no validation frame may occur
in probe optimization.

## Exact Target

For each action token, layer, and teacher view, extract the gated residual
contributed by that view through the teacher's actual attention operation:

`C_l^v = gate_l * W_O(sum_{j in v} P_l[:, j] V_l[j])`.

`P_l` is the normal full softmax over every valid image, language, and action
key. It is not renormalized over image tokens. `W_O` is the action expert's real
output projection. Agentview and wrist contributions are retained separately.

The negative control keeps the same teacher query, noisy action, timestep, and
all non-target keys. It rolls only one view's K/V tensors by one batch item and
recomputes the full softmax. This prevents shared action inputs from changing
between the correct and shuffled targets.

Before the real run, a unit test must partition all keys and verify that the sum
of extracted source contributions reconstructs the normal gated action-attention
residual with relative error below `1e-3`.

## Metrics

Report each layer and view separately, then their unweighted mean:

- held-out cosine between probe prediction and the correct contribution;
- cosine against the same-query shuffled-view contribution and their gap;
- held-out explained variance against the validation-mean baseline;
- target standard deviation and visual-to-total attention-output norm ratio;
- the same metrics on the top 25% of validation samples ranked by
  `student flow error - teacher flow error`;
- an episode-bootstrap 95% confidence interval for both cosine gaps.

## Decision

A layer is usable only if the three-seed mean satisfies all of these conditions:

- overall cosine gap is at least `0.10` and its bootstrap lower bound is positive;
- hard-subset gap is at least `0.05` and its bootstrap lower bound is positive;
- explained variance is positive;
- target variance, contribution norm, and gradients are finite and nonzero.

Select a layer only when its overall gap exceeds the runner-up by at least
`0.02`; otherwise treat passing layers as indistinguishable and use layer 6.

If a layer passes, the next experiment is a matched 5K Flow-only versus exact
visual-contribution distillation run at that one layer. If only the overall set
passes, add backview history before policy distillation because the hard samples
contain the missing information of interest. If no layer passes, stop hidden
feature matching and test teacher-guided hard-sample weighting rather than
tuning another cue-loss coefficient.

