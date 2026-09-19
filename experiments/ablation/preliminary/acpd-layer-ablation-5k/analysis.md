# H3 Analysis: ACPD Layer Ablation at 5K Steps

## Result

H3 is not supported. Single-layer ACPD at layer 6 has the lowest supervised
loss in both pre-registered windows, although all differences are below 1%.

| Layers | Mean loss, steps 100-4,900 | Mean loss, steps 4,000-4,900 | Final loss |
|---|---:|---:|---:|
| 6 | 0.040565 | 0.032250 | 0.0314 |
| 12 | 0.040882 | 0.032570 | 0.0317 |
| 6, 12 | 0.040682 | 0.032370 | 0.0317 |

Relative to layer 6, layers 6+12 are 0.29% worse over the primary window and
0.37% worse over the final window. The additional layer-12 head therefore does
not improve early convergence enough to justify its extra complexity. This
single-seed loss comparison does not establish task success.

## Decision

Prefer layer 6 for future ACPD experiments unless the saved two-layer policy
shows a task-success advantage. Evaluate the existing two-layer checkpoint
before launching another full-length layer configuration.
