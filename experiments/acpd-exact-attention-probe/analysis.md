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
selects layer 9 for H9. Layer 6 fails because explained variance is negative.

Raw result:
`/opt/liutong/openpi-5090-research/acpd-exact-attention-probe/results/metrics_128248.json`.
