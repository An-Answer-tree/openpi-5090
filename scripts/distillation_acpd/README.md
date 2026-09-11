# ACPD LoRA Distillation on RTX 5090

This implementation follows the method in `AAAI_2027_ACPD.pdf`. A frozen
agent-view and wrist-view pi0.5 teacher supervises an independently trained
fixed-view LoRA student. The selector and predictor are optimized jointly;
stop-gradient applies to teacher features and the student selector query, not
to selector parameters.

The loss is:

```text
L = Lflow + 0.2 * (Lpred + 0.1 * Lvar) + 0.5 * LACL
```

Each four-GPU job uses a physical global batch of 32 with no gradient
accumulation. Checkpoints are written below
`/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_lora/`.

The saved `params` are inference-compatible with the corresponding fixed-view
LoRA config; ACPD auxiliary heads are ignored by the normal policy loader.

Submit one student per job:

```bash
sbatch scripts/train_slurm/pi05_libero_backview_acpd_lora_fsdp4.sbatch
sbatch scripts/train_slurm/pi05_libero_topview_acpd_lora_fsdp4.sbatch
sbatch scripts/train_slurm/pi05_libero_leftview_acpd_lora_fsdp4.sbatch
sbatch scripts/train_slurm/pi05_libero_rightview_acpd_lora_fsdp4.sbatch
```
