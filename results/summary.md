# Results

Mean ± std over 3 seed(s). Shortcut gap = AUC(aligned) − AUC(flipped). ΔTPR = sensitivity on pneumonia images with the marker minus without it (balanced test set). Grad-CAM enrichment = share of 'pneumonia' heatmap energy in the marker region ÷ that region's share of the image (1.0 = ignored).

## Experiment 1 – shortcut strength sweep (ERM)

| Training data | Clean AUC | Aligned AUC | Flipped AUC | Shortcut gap | Worst-group acc | ΔTPR | Grad-CAM enrichment |
|---|---|---|---|---|---|---|---|
| No shortcut (clean training) | 0.985 ± 0.003 | 0.984 ± 0.004 | 0.985 ± 0.003 | -0.001 ± 0.002 | 0.830 ± 0.025 | -0.006 ± 0.005 | 0.738 ± 0.343 |
| ρ = 50% | 0.990 ± 0.004 | 0.990 ± 0.003 | 0.990 ± 0.004 | 0.000 ± 0.001 | 0.779 ± 0.020 | 0.000 ± 0.000 | 0.943 ± 0.367 |
| ρ = 70% | 0.988 ± 0.002 | 0.989 ± 0.002 | 0.974 ± 0.007 | 0.015 ± 0.006 | 0.757 ± 0.030 | 0.004 ± 0.000 | 1.789 ± 0.728 |
| ρ = 90% | 0.988 ± 0.002 | 0.993 ± 0.001 | 0.909 ± 0.026 | 0.085 ± 0.025 | 0.536 ± 0.019 | 0.024 ± 0.015 | 3.380 ± 0.500 |
| ρ = 100% | 0.944 ± 0.006 | 1.000 ± 0.000 | 0.008 ± 0.002 | 0.992 ± 0.002 | 0.011 ± 0.011 | 0.702 ± 0.087 | 6.914 ± 0.951 |

## Experiment 2 – mitigations at ρ = 90%

| Method | Clean AUC | Aligned AUC | Flipped AUC | Shortcut gap | Worst-group acc | ΔTPR | Grad-CAM enrichment |
|---|---|---|---|---|---|---|---|
| Clean reference (no shortcut) | 0.985 ± 0.003 | 0.984 ± 0.004 | 0.985 ± 0.003 | -0.001 ± 0.002 | 0.830 ± 0.025 | -0.006 ± 0.005 | 0.738 ± 0.343 |
| ERM (no mitigation) | 0.988 ± 0.002 | 0.993 ± 0.001 | 0.909 ± 0.026 | 0.085 ± 0.025 | 0.536 ± 0.019 | 0.024 ± 0.015 | 3.380 ± 0.500 |
| Group-balanced sampling | 0.989 ± 0.001 | 0.995 ± 0.001 | 0.961 ± 0.009 | 0.034 ± 0.009 | 0.826 ± 0.012 | 0.143 ± 0.011 | 3.019 ± 0.597 |
| Marker randomisation | 0.991 ± 0.001 | 0.990 ± 0.002 | 0.991 ± 0.001 | -0.001 ± 0.002 | 0.821 ± 0.033 | -0.008 ± 0.002 | 1.357 ± 0.512 |
| JTT | 0.986 ± 0.002 | 0.993 ± 0.001 | 0.920 ± 0.009 | 0.073 ± 0.009 | 0.580 ± 0.060 | 0.041 ± 0.011 | 3.132 ± 0.329 |
