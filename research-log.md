# Research Log

| # | Date | Type | Summary |
|---|---|---|---|
| 1 | 2026-09-11 | bootstrap | Compared the ACPD method in `AAAI_2027_ACPD.pdf` with the historical V6.4 implementation. The paper jointly trains the selector and predictor and uses cue variance regularization; V6.4 stops selector gradients and omits that regularizer. Locked H1 for a paper-faithful LoRA/FSDP4 implementation. |
| 2 | 2026-09-11 | result | Backview smoke job 126759 completed two optimizer steps with global micro-batch 8, accumulation 4, effective batch 32, finite losses, nonzero selector/predictor/LoRA gradients, and 17,291 MiB sampled peak GPU memory per card. Physical global batch 32 smoke job 126936 also completed two steps with the same checks and 17,337 MiB sampled peak memory per card. Selected physical batch 32, accumulation 1 for the full runs. |
| 3 | 2026-09-12 | protocol | Cancelled the four pending 30K view-specific jobs before allocation. Locked H3: compare backview ACPD at layers 6, 12, and 6+12 for 5K steps using matched initialization, seed, data, and optimization. Primary metric is mean supervised loss over logged steps 100-4,900; the final-window metric covers steps 4,000-4,900. |
