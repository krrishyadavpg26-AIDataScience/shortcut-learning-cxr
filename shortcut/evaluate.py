"""Evaluation on clean / aligned / flipped / balanced test sets."""
import numpy as np

from .data import GROUP_NAMES, ShortcutDataset, group_ids, test_markers
from .train import predict, safe_auc

# name -> marker mode used to build that test set
TEST_SETS = {
    "clean": "none",          # no markers: true disease signal only
    "aligned": "correlated",  # same marker-label correlation as training
    "flipped": "flipped",     # marker only on normal images: shortcut reversed
    "balanced": "random",     # marker on a random 50%: for group metrics
}


def binary_metrics(y, p, thr=0.5):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= thr).astype(int)
    pos, neg = y == 1, y == 0
    return {
        "auc": safe_auc(y, p),
        "acc": float((pred == y).mean()),
        "tpr": float(pred[pos].mean()) if pos.any() else float("nan"),   # sensitivity
        "tnr": float(1 - pred[neg].mean()) if neg.any() else float("nan"),  # specificity
    }


def evaluate_model(model, test_imgs, test_labels, size, eval_rho, device, num_workers=0):
    """eval_rho sets the correlation of the 'aligned' test set (the training rho)."""
    out = {}
    for name, mode in TEST_SETS.items():
        markers = test_markers(test_labels, mode, eval_rho)
        ds = ShortcutDataset(test_imgs, test_labels, markers, size)
        p, y, m = predict(model, ds, device, num_workers=num_workers)
        met = binary_metrics(y, p)
        met["marker_rate"] = float(m.mean())
        if name == "balanced":
            pred = (p >= 0.5).astype(int)
            g = group_ids(y, m)
            accs = {GROUP_NAMES[k]: float((pred[g == k] == y[g == k]).mean())
                    for k in range(4) if (g == k).any()}
            met["group_acc"] = accs
            met["worst_group_acc"] = float(min(accs.values()))
            # Delta TPR between bias-aligned (pneumonia+marker) and bias-contrasting
            # (pneumonia without marker) positives, as in Boland et al. 2025.
            met["delta_tpr"] = accs.get("pneumonia_marker", np.nan) - accs.get(
                "pneumonia_nomarker", np.nan)
        out[name] = met
    out["shortcut_gap"] = out["aligned"]["auc"] - out["flipped"]["auc"]
    return out
