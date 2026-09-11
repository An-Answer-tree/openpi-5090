# H1 Protocol: ACPD LoRA on Four RTX 5090 GPUs

## Hypothesis

Paper-faithful ACPD can complete training steps on four 32 GB RTX 5090 GPUs using a LoRA pi0.5 student, a frozen full pi0.5 teacher, FSDP4, global micro-batch 8, and four gradient accumulation steps. The effective global batch size is 32.

## Fixed Method

- Dataset: `/opt/liutong/libero_multiview_tuned_6view_lerobot`.
- Teacher: agent view plus wrist view from `/opt/liutong/openpi_checkpoints/fixed_dataset/sft/agentview_wrist/29999/params`.
- Students: independent back, top, left, and right fixed-view policies initialized from base pi0.5.
- Action horizon: 10.
- ACPD layers: 6 and 12; cue dimension: 1024.
- Loss: `Lflow + 0.2 * (Lpred + 0.1 * Lvar) + 0.5 * LACL`.
- Selector query and teacher features use stop-gradient; selector and predictor parameters remain trainable.
- LR: 1,000-step warmup to 2.5e-5, cosine decay to 2.5e-6 at step 30,000.
- Student LoRA: PaliGemma rank 16 and action expert rank 32 using the repository defaults.
- EMA disabled; AdamW gradient clipping norm 1.0.

## Memory Plan

- Four-device FSDP for student parameters, optimizer state, and frozen teacher parameters.
- Global micro-batch 8, or 2 samples per GPU, with four accumulated micro-steps.
- Existing transformer rematerialization and bfloat16 frozen parameters.
- One student view per Slurm job.

## Confirmatory Smoke Test

Run backview for two optimizer steps. The hypothesis is supported for launch feasibility when:

- compilation and both optimizer steps complete without host or device OOM;
- total loss, flow loss, cue prediction loss, cue variance loss, and ACL are finite;
- ACPD selector and predictor gradient norms are nonzero;
- logs report global micro-batch 8, accumulation 4, and effective batch size 32;
- observed peak memory stays below the 32 GB device limit.

If this configuration fails from device OOM, reduce the global micro-batch to 4 and use eight accumulation steps. If it succeeds with substantial headroom, test global micro-batch 16 with two accumulation steps as an exploratory throughput optimization.
