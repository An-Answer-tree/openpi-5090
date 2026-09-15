# H9 Protocol: Recoverability-gated ACPD-v2 5K Screen

## Question

Can privileged information improve a backview policy beyond the matched
ACL-only baseline when the transfer mechanism follows H7's recoverability
test?

## Branch Selection

Select exactly one branch from the completed H7 JSON before implementing or
submitting H9:

- If `selected_layer` is not null, run exact-contribution distillation at that
  layer.
- Otherwise, run teacher-advantage sample weighting. A layer that passes only
  the overall set but not the pre-registered hard subset is not eligible for
  feature matching under the current no-expansion constraint.

Do not run both branches.

## Exact-contribution Branch

At H7's selected action-expert layer, predict the teacher's separate
agentview and wrist attention residuals from the student action hidden state.
Use the same linear decoder class validated by H7 and normalized MSE:

`L_cue = MSE(prediction, stopgrad(target)) / mean(target^2)`.

Sum the two predicted residuals and add them to the student's final action
representation through a trainable scalar `tanh(alpha)`, with `alpha=0` at
initialization. Stop the flow-loss gradient through the predicted residual;
the predictor and its student input are trained only by `L_cue`, while the gate
is trained by flow matching. Keep the predictor and gate in the checkpoint and
use both during policy inference.

The objective is:

`L = L_flow + 0.2 L_cue + 0.5 L_ACL`.

## Teacher-advantage Branch

If H7 selects no layer, do not train another cue predictor. For each sample,
compute task-dimension flow errors `E_s` and `E_t`, then use:

`d = relu(E_s - E_t)`

`w = (1 + d / mean(d)) / 2`.

Stop gradients through `w`. Its batch mean is one and its minimum is 0.5.
Optimize `mean(w * L_flow_per_sample) + 0.5 L_ACL`. Inference uses the normal
student model without the teacher.

## Shared Training Control

- Student view: backview.
- Teacher: agentview+wrist step 29,999.
- Dataset: `libero_multiview_tuned_6view_lerobot` at its original path.
- Seed: 42; optimizer steps: 30,000 in one continuous run.
- Effective global batch: 32.
- Schedule: 1K warmup, cosine `2.5e-5` to `2.5e-6` over 30K steps.
- Hardware: four-device FSDP. The exact branch may use physical batch 8 with
  four accumulation steps; the weighted branch uses physical batch 32.
- Save every 5,000 steps and keep all six checkpoints through step 29,999. Do
  not copy data or checkpoints.

## Execution Amendment

This amendment was locked after job 128513 failed before step 0 and before any
successful H9 training result existed. The first 5,000 steps remain identical
to the original H9 screen. Training continues to 30,000 steps in the same job
to avoid a later resume queue. A one-CPU watcher submits the existing four-GPU
evaluation only after checkpoint 4,999 contains finalized `params`, `assets`,
and `_CHECKPOINT_METADATA`; it does not reserve evaluation GPUs while waiting.

The pre-registered H9 decision is still made only at step 4,999 against the
matched 5K H8 checkpoint. Later checkpoints are a pre-planned exploratory
continuation and require a duration-matched baseline before any claim about
ACPD efficacy at 30K.

## Evaluation and Decision

Evaluate the selected H9 checkpoint on the same fixed backview camera, four
LIBERO suites, 50 trials per task, seed 7, and replan interval 5. Compare all
2,000 paired episode outcomes with H8 ACL-only using 10,000 task-stratified
bootstrap samples.

H9 passes only if it exceeds ACL-only pooled success by at least 1.5 percentage
points and the paired 95% confidence interval excludes zero. Also report
Flow-only and old Full ACPD as fixed references. This screen includes no
repeated seed, longer run, or additional student view.
