# Research Findings

## Research Question

Can paper-faithful ACPD distillation train a LoRA pi0.5 student with effective batch size 32 on four 32 GB RTX 5090 GPUs?

## Current Understanding

The student can use the repository's existing pi0.5 LoRA variants. LoRA reduces student gradient and optimizer memory, but the frozen full teacher and ACPD auxiliary heads remain resident. The stable configuration combines FSDP4, rematerialization, no EMA, and a physical global batch of 32.

## Key Results

Two backview 5090 smoke runs completed successfully. Job 126759 used global micro-batch 8 with four accumulation steps; job 126936 used physical global batch 32 with no accumulation. Both completed two optimizer steps with finite losses and nonzero selector, predictor, and LoRA gradients. Peak sampled GPU memory was 17,291 MiB/card and 17,337 MiB/card, respectively.

## Patterns and Insights

The paper method requires gradients through the cue selector while stopping gradients only through the student query and teacher features. Historical V6.4 is not paper-faithful because it stops gradients through the selected cue and omits the variance term.

The exact Table 1 ACPD scores are reproduced by the old V6.3 `historical_sg` logs. That implementation deliberately stops the selected cue and omits the variance regularizer, unlike Equations 2-5 in the manuscript. Table 2 does come from the joint-selector V6 implementation and provides genuine positive task-success evidence for that version under full-parameter fine-tuning.

At 5K LoRA steps, layers 12 and 6+12 have essentially the same supervised-loss convergence. Their final-window means are 0.03257 and 0.03237, respectively, so the dual-layer improvement is only 0.61% and does not meet the pre-registered 1% rule. The cue prediction also becomes easy rapidly: its weighted loss falls from more than twice the flow loss at initialization to about 2% of the flow loss by step 4,900. This is compatible with either successful representation alignment or selector-predictor co-adaptation; loss values alone cannot distinguish them.

The next controlled test isolates flow-only, Cue-only, and ACL-only optimization within the same trainer and RNG schedule. Parameter tuning is deferred until this component test identifies which term changes student learning.

## Lessons and Constraints

- Train one student view per four-GPU job; do not place four teacher-student pairs in one allocation.
- Effective batch size is `global micro-batch * gradient accumulation steps`.
- Teacher visual and action features depend on the noisy action and flow time, so they cannot be fully precomputed without changing the method.
- Main-table provenance must be corrected or rerun: V6.3 historical stop-gradient results cannot be described as the joint-selector, variance-regularized method.

## Open Questions

- Do the four 30K view-specific runs remain stable after the smoke configuration is promoted?
- Does checkpoint writing remain the dominant wall-clock cost at the configured save interval?
- Is the teacher more accurate than the student on the exact sampled flow targets throughout LoRA training?
- Does Cue or ACL improve task success even when supervised flow loss is unchanged?

## Optimization Trajectory

The physical global batch 32 run is preferred because it uses the complete batch for the variance statistic and removes unnecessary accumulation steps. The sampled peak was about 16.9 GiB/card, leaving about 15.1 GiB before the nominal 32 GiB device limit.

The dual-layer head is not justified by the completed early-loss comparison unless the pending layer-6 run is worse than layer 12 and later task evaluation shows a benefit. The current default for follow-up diagnostics remains layers 6+12 only to reuse the completed full-objective trajectory as a matched control.
