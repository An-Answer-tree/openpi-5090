# Research Findings

## Research Question

How can privileged `agentview+wrist` information train a stronger weak-view
pi0.5 student?

## Current Understanding

The student can use the repository's existing pi0.5 LoRA variants. LoRA reduces student gradient and optimizer memory, but the frozen full teacher and ACPD auxiliary heads remain resident. The stable configuration combines FSDP4, rematerialization, no EMA, and a physical global batch of 32.

The layer-10 ACPD-v2 policy also passes the physical-global-batch-32 runtime
gate on four RTX 5090 GPUs without gradient accumulation. Its first four
optimizer steps are finite with nonzero LoRA, predictor, and residual-gate
gradients; peak sampled memory is 17,402 MiB per card. This establishes runtime
feasibility, not task-success efficacy.

## Key Results

Two backview 5090 smoke runs completed successfully. Job 126759 used global micro-batch 8 with four accumulation steps; job 126936 used physical global batch 32 with no accumulation. Both completed two optimizer steps with finite losses and nonzero selector, predictor, and LoRA gradients. Peak sampled GPU memory was 17,291 MiB/card and 17,337 MiB/card, respectively.

The trainer-matched 2K component ablation does not support an early-convergence effect at the pre-registered 1% resolution. Relative to Flow-only, Cue-only changed the primary and final-window supervised losses by +0.80% and +0.98%, ACL-only by -0.31% and -0.63%, and Full ACPD by +0.11% and +0.16%. All four runs completed with physical batch 32 on two-device FSDP at about 31.4 GiB peak memory per card.

The 5K layer ablation also does not support the dual-layer hypothesis. Layer 6, layer 12, and layers 6+12 have primary losses of 0.040565, 0.040882, and 0.040682, respectively. Layer 6 is also best in the final window at 0.032250, versus 0.032570 and 0.032370. The matched 5K Flow-only policy achieved 89/2,000 successes (4.45% pooled), while Full ACPD achieved 130/2,000 (6.50%). The +2.05-point difference meets the pre-registered effect threshold; its task-stratified paired-bootstrap 95% confidence interval is [+0.90, +3.25] points.

## Patterns and Insights

The paper method requires gradients through the cue selector while stopping gradients only through the student query and teacher features. Historical V6.4 is not paper-faithful because it stops gradients through the selected cue and omits the variance term.

The exact Table 1 ACPD scores are reproduced by the old V6.3 `historical_sg` logs. That implementation deliberately stops the selected cue and omits the variance regularizer, unlike Equations 2-5 in the manuscript. The exact SFT scores come from the corresponding old full-parameter `libero_multiview` runs, not the current fixed dataset.

Table 2 uses the joint-selector V6 implementation, but it is not a controlled ablation. The reported Cue-only A row is ungated, Cue+ACL A is gated, and Cue+ACL A+W is ungated. The manuscript does not define or disclose the teacher-reliability gate. Consequently, Table 2 is evidence that some V6 configurations performed above SFT, but its row differences cannot isolate the effects of ACL or the wrist view.

The manuscript's learning-rate description also differs from the old launchers. Both SFT and ACPD inherit a 10K warmup to `5e-5`, followed by a schedule whose peak and final rates are both `5e-5`; it is constant after warmup rather than cosine-decayed. The current LoRA experiment uses a true 1K-warmup, 30K cosine decay from `2.5e-5` to `2.5e-6`. This is a deliberate LoRA engineering choice, not a reproduction of the manuscript schedule, and has not yet been tuned for ACPD.

At 5K LoRA steps, single layer 6 has the lowest primary and final-window supervised losses. Layers 6+12 are 0.29% and 0.37% worse in those windows, so the extra head is not justified by early convergence. The cue prediction also becomes easy rapidly: its weighted loss falls from more than twice the flow loss at initialization to about 2% of the flow loss by step 4,900. This is compatible with either successful representation alignment or selector-predictor co-adaptation; loss values alone cannot distinguish them.

The component test confirms that both auxiliary branches are active without establishing an optimization advantage. Cue-only retains nonzero selector and predictor gradients and reaches 0.996 cue cosine without variance collapse by step 1,900. ACL nearly doubles the mean LoRA gradient norm and slightly lowers action-correlation loss. Despite these changes, Full ACPD is nearly indistinguishable from Flow-only on supervised convergence. This is consistent with either weak opposing component effects or auxiliary changes that are not visible in the training-loss proxy.

The teacher is better than the student on about 99.9% of sampled task-dimension flow targets during the 2K runs, with a primary-window MSE of 0.01595 versus about 0.229 for the student. Teacher reliability is therefore not the immediate bottleneck, and the historical reliability gate is not justified by this diagnostic. The next useful discriminator is task success from matched saved checkpoints, not a sweep over Cue and ACL weights.

H5 shows that the full objective can improve task success even when supervised
loss is insensitive. H8 isolates this gain: ACL-only is statistically better
than Flow-only and indistinguishable from Full ACPD under the locked screening
rule. The H5 result therefore does not provide evidence for the old Cue branch.

The next cue branch replaces the learned or heuristic cue with the teacher's exact
per-view attention residual: real Q/K/V projections, the full attention
softmax, the action expert output projection, and the AdaRMS residual gate. A
same-query control changes only one view's K/V tensors. This directly tests
whether privileged visual information is predictable from backview, including
the samples where the teacher has the largest flow-error advantage.

H7 validates this target on 256 episode-held-out samples. Layer 9 has an
overall correct-minus-shuffled cosine gap of 0.3417 (95% CI [0.3095, 0.3756]),
a teacher-advantage hard-subset gap of 0.3194 ([0.2717, 0.3875]), and explained
variance of 0.2822. Layer 12 also passes, but its overall gap is 0.2367; the
0.1050 margin exceeds the pre-registered 0.02 rule, so H9 uses only layer 9.
Layer 6 fails because its explained variance is -0.1468.

H8 completed the missing component attribution. ACL-only reached 127/2,000
successes (6.35%), compared with 89/2,000 (4.45%) for Flow-only and 130/2,000
(6.50%) for Full ACPD. ACL-only improves over Flow-only by 1.90 points with a
paired 95% CI of [+0.75, +3.10]. Full ACPD improves over ACL-only by only 0.15
points with CI [-1.15, +1.40]. Under the locked 0.5-point rule, ACL explains
the H5 gain at this screening resolution; the old Cue has no detected
incremental pooled benefit.

H7.1 completed the local layer scan under the unchanged H7 protocol. Layers
7, 8, 10, and 11 all pass the recoverability gates. Layer 10 has the largest
overall gap at 0.3992 (95% CI [0.3524, 0.4312]), hard gap 0.3779, and explained
variance 0.3556. It exceeds runner-up layer 11 by 0.0485, so the combined
layer-6-through-12 scan selects layer 10. This refutes layer 9 as the stable
local recoverability optimum but does not measure policy success.

## Lessons and Constraints

- Train one student view per four-GPU job; do not place four teacher-student pairs in one allocation.
- Effective batch size is `global micro-batch * gradient accumulation steps`.
- Teacher visual and action features depend on the noisy action and flow time, so they cannot be fully precomputed without changing the method.
- Main-table provenance must be corrected or rerun: V6.3 historical stop-gradient results cannot be described as the joint-selector, variance-regularized method.
- Table 2 must use one gate policy across every row, or explicitly include the gate as an ablation factor.
- The paper must report the schedule that produced its tables or rerun with the stated cosine schedule.
- Early supervised-loss convergence is too insensitive to select ACPD components; use it for sanity checks and use benchmark success for method decisions.
- A privileged target must be fixed independently of the predictor and compared
  with a control that preserves noisy action and timestep; otherwise shared
  action inputs can masquerade as transferred visual information.

## Open Questions

- Does checkpoint writing remain the dominant wall-clock cost at the configured save interval?
- Does the teacher's strong early flow-target advantage persist later in training?
- Does deploying the layer-10 exact-contribution predictor improve task success
  beyond ACL-only at the same 5K budget?

## Optimization Trajectory

The physical global batch 32 run is preferred because it uses the complete batch for the variance statistic and removes unnecessary accumulation steps. The sampled peak was about 16.9 GiB/card, leaving about 15.1 GiB before the nominal 32 GiB device limit.

The dual-layer head is not justified by the completed early-loss comparison. Layer 6 is the preferred configuration for new experiments, while the existing layers 6+12 checkpoint remains the matched Full ACPD control for the pending task-success evaluation.

H4 rules out supervised-loss weight tuning as the next step. H5 passes its
task-success threshold. The matched H8 evaluation attributes that gain to ACL
at the locked screening resolution; the old Cue should not be promoted without
new evidence.

The coarse H7 scan selected layer 9 before H7.1 selected layer 10 across layers
6 through 12. H9 therefore tests only the best recoverable layer. It uses the
H8 ACL weight, physical global batch 32 without accumulation, separate
agentview and wrist contribution prediction, and a deployed gated residual.
Task-success results are pending.
