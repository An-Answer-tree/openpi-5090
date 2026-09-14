# ACPD LoRA Distillation on RTX 5090

A frozen agent-view and wrist-view pi0.5 teacher supervises an independently
trained fixed-view LoRA student. The selector and predictor are optimized
jointly; stop-gradient applies to teacher features and the student selector
query, not to selector parameters.

The loss is:

```text
L = Lflow + 0.2 * (Lpred + 0.1 * Lvar) + 0.5 * LACL
```

Controlled experiments use a physical global batch of 32. Checkpoints are
written below
`/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_lora/`.

The saved `params` are inference-compatible with the corresponding fixed-view
LoRA config; ACPD auxiliary heads are ignored by the normal policy loader.
Use the pre-registered 2K/5K launchers in `scripts/train_slurm/` and record
verified results in `EXPERIMENT_RESULTS.md`. The original four view-specific
30K launchers were retired because the controlled ablations did not justify
long runs of that objective.
