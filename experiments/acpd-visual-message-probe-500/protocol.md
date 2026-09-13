# H6 Protocol: Visual-Message Probe at 500 Steps

## Question

Can a backview student predict the part of the teacher action representation
that comes from the teacher's privileged images, without using teacher action
tokens as a shortcut?

## Motivation

The existing ACPD selector reaches near-perfect cue cosine, but its attention
uses teacher action tokens heavily. Those tokens contain the same noisy action
and flow time as the student, so the loss can be solved without transferring
privileged visual information. This probe removes that shortcut.

## Experiment

- Teacher input: `agentview_image` plus `wrist_image`.
- Student input: `backview_image` only.
- Extract only the teacher visual attention message received by each action
  token at layer 6. Do not include teacher action hidden states.
- Keep the teacher and student frozen. Train only a linear probe from the
  student layer-6 action hidden state to the teacher visual message.
- Use fixed dataset `libero_multiview_tuned_6view_lerobot`, seed 42, physical
  global batch 32, two-device FSDP, and 500 optimizer steps.
- Do not save a model checkpoint.

## Metrics and Decision

At every logged checkpoint, compare cosine similarity for the correct teacher
message with cosine similarity after shuffling teacher messages within the
batch. Also report teacher-message variance and probe gradient norm.

The probe passes only when the final 100-step mean of

`correct_cosine - shuffled_cosine`

is at least `0.10`, while teacher-message variance is nonzero and gradients are
finite. This is a feasibility screen, not a policy result.

If it passes, implement visual-only ACPD and run a matched 5K backview policy
comparison. If it fails, stop feature matching and test a teacher image-on vs.
image-masked action residual target instead. If that also fails, test a short
student observation history rather than further loss-weight tuning.

