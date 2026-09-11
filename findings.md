# Research Findings

## Research Question

Can paper-faithful ACPD distillation train a LoRA pi0.5 student with effective batch size 32 on four 32 GB RTX 5090 GPUs?

## Current Understanding

The student can use the repository's existing pi0.5 LoRA variants. LoRA reduces student gradient and optimizer memory, but the frozen full teacher and ACPD auxiliary heads remain resident. The initial stable configuration therefore combines FSDP4, rematerialization, no EMA, and a global micro-batch of 8 accumulated four times.

## Key Results

No 5090 ACPD run has been measured yet.

## Patterns and Insights

The paper method requires gradients through the cue selector while stopping gradients only through the student query and teacher features. Historical V6.4 is not paper-faithful because it stops gradients through the selected cue and omits the variance term.

## Lessons and Constraints

- Train one student view per four-GPU job; do not place four teacher-student pairs in one allocation.
- Effective batch size is `global micro-batch * gradient accumulation steps`.
- Teacher visual and action features depend on the noisy action and flow time, so they cannot be fully precomputed without changing the method.

## Open Questions

- Does global micro-batch 8 leave enough memory headroom on each RTX 5090?
- If it does, can global micro-batch 16 with two accumulation steps improve throughput?

## Optimization Trajectory

No completed runs.
