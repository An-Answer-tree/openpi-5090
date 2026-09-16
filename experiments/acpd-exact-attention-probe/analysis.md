# H7 Result: Exact Privileged-attention Recoverability

Validation used 256 samples from held-out episodes; the teacher-advantage hard
subset contained 64 samples.

| Layer | Overall gap | 95% CI | Hard gap | Hard 95% CI | Explained variance | Usable |
|---:|---:|---|---:|---|---:|---|
| 6 | 0.0997 | [0.0846, 0.1091] | 0.0706 | [0.0497, 0.0884] | -0.1468 | No |
| **9** | **0.3417** | **[0.3095, 0.3756]** | **0.3194** | **[0.2717, 0.3875]** | **0.2822** | **Yes** |
| 12 | 0.2367 | [0.2127, 0.2546] | 0.2080 | [0.1692, 0.2349] | 0.0903 | Yes |

Layer 9 exceeds layer 12 by 0.1050 overall cosine gap, above the locked 0.02
selection margin. H7 therefore supports exact-contribution recoverability and
selected layer 9 for the already locked H9 experiment. Layer 6 fails because
explained variance is negative.

Raw result:
`/opt/liutong/openpi-5090-research/acpd-exact-attention-probe/results/metrics_128248.json`.

## H7.1 Local Layer Scan

H7.1 reused the same split, controls, probe seeds, optimization budget, and
decision thresholds. It added layers 7, 8, 10, and 11.

| Layer | Overall gap | 95% CI | Hard gap | Hard 95% CI | Explained variance | Usable |
|---:|---:|---|---:|---|---:|---|
| 7 | 0.3261 | [0.2938, 0.3602] | 0.2776 | [0.2300, 0.3390] | 0.2833 | Yes |
| 8 | 0.2858 | [0.2491, 0.3030] | 0.2373 | [0.1970, 0.2794] | 0.3420 | Yes |
| **10** | **0.3992** | **[0.3524, 0.4312]** | **0.3779** | **[0.3235, 0.4578]** | **0.3556** | **Yes** |
| 11 | 0.3508 | [0.3170, 0.3824] | 0.3045 | [0.2435, 0.3634] | 0.3113 | Yes |

Across layers 6 through 12, layer 10 exceeds runner-up layer 11 by 0.0485
overall gap, above the locked 0.02 margin. H7.1 therefore rejects layer 9 as
the stable local optimum and selects layer 10 for future policy experiments.
It does not change the in-flight layer-9 H9 protocol and does not measure task
success.

Raw result:
`/opt/liutong/openpi-5090-research/acpd-exact-attention-probe/results/metrics_128789.json`.
