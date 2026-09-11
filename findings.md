# Research Findings

## Research Question

Can paper-faithful ACPD distillation train a LoRA pi0.5 student with effective batch size 32 on four 32 GB RTX 5090 GPUs?

## Current Understanding

The student can use the repository's existing pi0.5 LoRA variants. LoRA reduces student gradient and optimizer memory, but the frozen full teacher and ACPD auxiliary heads remain resident. The stable configuration combines FSDP4, rematerialization, no EMA, and a physical global batch of 32.

## Key Results

Two backview 5090 smoke runs completed successfully. Job 126759 used global micro-batch 8 with four accumulation steps; job 126936 used physical global batch 32 with no accumulation. Both completed two optimizer steps with finite losses and nonzero selector, predictor, and LoRA gradients. Peak sampled GPU memory was 17,291 MiB/card and 17,337 MiB/card, respectively.

## Patterns and Insights

The paper method requires gradients through the cue selector while stopping gradients only through the student query and teacher features. Historical V6.4 is not paper-faithful because it stops gradients through the selected cue and omits the variance term.

## Lessons and Constraints

- Train one student view per four-GPU job; do not place four teacher-student pairs in one allocation.
- Effective batch size is `global micro-batch * gradient accumulation steps`.
- Teacher visual and action features depend on the noisy action and flow time, so they cannot be fully precomputed without changing the method.

## Open Questions

- Do the four 30K view-specific runs remain stable after the smoke configuration is promoted?
- Does checkpoint writing remain the dominant wall-clock cost at the configured save interval?

## Optimization Trajectory

The physical global batch 32 run is preferred because it uses the complete batch for the variance statistic and removes unnecessary accumulation steps. The sampled peak was about 16.9 GiB/card, leaving about 15.1 GiB before the nominal 32 GiB device limit.
