# Results

Mean ± std over 3 seed(s). Shortcut gap = AUC(aligned) − AUC(flipped). ΔTPR = sensitivity on pneumonia images with the marker minus without it (balanced test set). Grad-CAM enrichment = share of 'pneumonia' heatmap energy in the marker region ÷ that region's share of the image (1.0 = ignored).

## Experiment 1 – shortcut strength sweep (ERM)

| Training data | Clean AUC | Aligned AUC | Flipped AUC | Shortcut gap | Worst-group acc | ΔTPR | Grad-CAM enrichment |
|---|---|---|---|---|---|---|---|

## Experiment 2 – mitigations at ρ = 90%

| Method | Clean AUC | Aligned AUC | Flipped AUC | Shortcut gap | Worst-group acc | ΔTPR | Grad-CAM enrichment |
|---|---|---|---|---|---|---|---|
| JTT | 0.981 ± 0.005 | 0.991 ± 0.005 | 0.873 ± 0.015 | 0.118 ± 0.010 | 0.500 ± 0.089 | 0.026 ± 0.003 | 2.761 ± 0.400 |
