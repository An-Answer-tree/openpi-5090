# Research Findings

## Research Question

Can paper-faithful ACPD distillation train a LoRA pi0.5 student with effective batch size 32 on four 32 GB RTX 5090 GPUs?

## Current Understanding

The student can use the repository's existing pi0.5 LoRA variants. LoRA reduces student gradient and optimizer memory, but the frozen full teacher and ACPD auxiliary heads remain resident. The stable configuration combines FSDP4, rematerialization, no EMA, and a physical global batch of 32.

## Key Results

Two backview 5090 smoke runs completed successfully. Job 126759 used global micro-batch 8 with four accumulation steps; job 126936 used physical global batch 32 with no accumulation. Both completed two optimizer steps with finite losses and nonzero selector, predictor, and LoRA gradients. Peak sampled GPU memory was 17,291 MiB/card and 17,337 MiB/card, respectively.

The trainer-matched 2K component ablation does not support an early-convergence effect at the pre-registered 1% resolution. Relative to Flow-only, Cue-only changed the primary and final-window supervised losses by +0.80% and +0.98%, ACL-only by -0.31% and -0.63%, and Full ACPD by +0.11% and +0.16%. All four runs completed with physical batch 32 on two-device FSDP at about 31.4 GiB peak memory per card.

## Patterns and Insights

The paper method requires gradients through the cue selector while stopping gradients only through the student query and teacher features. Historical V6.4 is not paper-faithful because it stops gradients through the selected cue and omits the variance term.

The exact Table 1 ACPD scores are reproduced by the old V6.3 `historical_sg` logs. That implementation deliberately stops the selected cue and omits the variance regularizer, unlike Equations 2-5 in the manuscript. The exact SFT scores come from the corresponding old full-parameter `libero_multiview` runs, not the current fixed dataset.

Table 2 uses the joint-selector V6 implementation, but it is not a controlled ablation. The reported Cue-only A row is ungated, Cue+ACL A is gated, and Cue+ACL A+W is ungated. The manuscript does not define or disclose the teacher-reliability gate. Consequently, Table 2 is evidence that some V6 configurations performed above SFT, but its row differences cannot isolate the effects of ACL or the wrist view.

The manuscript's learning-rate description also differs from the old launchers. Both SFT and ACPD inherit a 10K warmup to `5e-5`, followed by a schedule whose peak and final rates are both `5e-5`; it is constant after warmup rather than cosine-decayed. The current LoRA experiment uses a true 1K-warmup, 30K cosine decay from `2.5e-5` to `2.5e-6`. This is a deliberate LoRA engineering choice, not a reproduction of the manuscript schedule, and has not yet been tuned for ACPD.

At 5K LoRA steps, layers 12 and 6+12 have essentially the same supervised-loss convergence. Their final-window means are 0.03257 and 0.03237, respectively, so the dual-layer improvement is only 0.61% and does not meet the pre-registered 1% rule. The cue prediction also becomes easy rapidly: its weighted loss falls from more than twice the flow loss at initialization to about 2% of the flow loss by step 4,900. This is compatible with either successful representation alignment or selector-predictor co-adaptation; loss values alone cannot distinguish them.

The component test confirms that both auxiliary branches are active without establishing an optimization advantage. Cue-only retains nonzero selector and predictor gradients and reaches 0.996 cue cosine without variance collapse by step 1,900. ACL nearly doubles the mean LoRA gradient norm and slightly lowers action-correlation loss. Despite these changes, Full ACPD is nearly indistinguishable from Flow-only on supervised convergence. This is consistent with either weak opposing component effects or auxiliary changes that are not visible in the training-loss proxy.

The teacher is better than the student on about 99.9% of sampled task-dimension flow targets during the 2K runs, with a primary-window MSE of 0.01595 versus about 0.229 for the student. Teacher reliability is therefore not the immediate bottleneck, and the historical reliability gate is not justified by this diagnostic. The next useful discriminator is task success from matched saved checkpoints, not a sweep over Cue and ACL weights.

## Lessons and Constraints

- Train one student view per four-GPU job; do not place four teacher-student pairs in one allocation.
- Effective batch size is `global micro-batch * gradient accumulation steps`.
- Teacher visual and action features depend on the noisy action and flow time, so they cannot be fully precomputed without changing the method.
- Main-table provenance must be corrected or rerun: V6.3 historical stop-gradient results cannot be described as the joint-selector, variance-regularized method.
- Table 2 must use one gate policy across every row, or explicitly include the gate as an ablation factor.
- The paper must report the schedule that produced its tables or rerun with the stated cosine schedule.
- gpu03 has a lost physical GPU and unreliable GPU isolation; ACPD jobs must exclude that node. This is an infrastructure failure, not evidence about the method or batch-size feasibility.
- Early supervised-loss convergence is too insensitive to select ACPD components; use it for sanity checks and use benchmark success for method decisions.

## Open Questions

- Do the four 30K view-specific runs remain stable after the smoke configuration is promoted?
- Does checkpoint writing remain the dominant wall-clock cost at the configured save interval?
- Does the teacher's strong early flow-target advantage persist later in training?
- Does Full ACPD improve task success over a trainer-matched Flow-only checkpoint even though their supervised losses are indistinguishable?

## Optimization Trajectory

The physical global batch 32 run is preferred because it uses the complete batch for the variance statistic and removes unnecessary accumulation steps. The sampled peak was about 16.9 GiB/card, leaving about 15.1 GiB before the nominal 32 GiB device limit.

The dual-layer head is not justified by the completed early-loss comparison unless the running layer-6 run is worse than layer 12 and later task evaluation shows a benefit. The current default for follow-up diagnostics remains layers 6+12 only to reuse the completed full-objective trajectory as a matched control.

H4 rules out supervised-loss weight tuning as the next step. The efficient next branch is sequential: first compare saved 5K Flow-only and Full ACPD policies on backview benchmarks; only if Full ACPD improves task success should Cue-only and ACL-only be promoted to checkpoint-producing runs.
