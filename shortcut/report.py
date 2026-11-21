"""Aggregate per-run JSON files into tables and figures."""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METHOD_LABELS = {
    "clean_ref": "Clean reference (no shortcut)",
    "erm": "ERM (no mitigation)",
    "group_balance": "Group-balanced sampling",
    "marker_random": "Marker randomisation",
    "jtt": "JTT",
}
COLORS = {"clean": "#2a7f62", "aligned": "#3b6fb6", "flipped": "#c8553d"}


def _flatten(rec):
    m = rec["metrics"]
    row = {k: rec[k] for k in ("name", "seed", "method", "time_sec")}
    row["rho"] = rec["rho"] if rec["rho"] is not None else np.nan
    for s in ("clean", "aligned", "flipped", "balanced"):
        for k in ("auc", "acc", "tpr", "tnr"):
            row[f"{s}_{k}"] = m[s][k]
    row["worst_group_acc"] = m["balanced"]["worst_group_acc"]
    row["delta_tpr"] = m["balanced"]["delta_tpr"]
    row["shortcut_gap"] = m["shortcut_gap"]
    row["mar"] = rec["gradcam"]["mar"]
    row["mar_enrichment"] = rec["gradcam"]["enrichment"]
    return row


KEY_COLS = ["clean_auc", "aligned_auc", "flipped_auc", "shortcut_gap",
            "worst_group_acc", "delta_tpr", "mar_enrichment"]


def load_runs(out_dir):
    recs = []
    for f in sorted(glob.glob(os.path.join(out_dir, "runs", "*.json"))):
        with open(f) as fh:
            recs.append(json.load(fh))
    return recs


def summarise(df):
    g = df.groupby(["name", "method", "rho"], dropna=False)
    mean, std = g[KEY_COLS].mean(), g[KEY_COLS].std(ddof=0)
    summ = mean.add_suffix("_mean").join(std.add_suffix("_std"))
    summ["n_seeds"] = g.size()
    return summ.reset_index()


def _fmt(row, col):
    m, s = row[f"{col}_mean"], row[f"{col}_std"]
    if pd.isna(m):
        return "–"
    m = 0.0 if abs(m) < 5e-4 else m  # avoid printing "-0.000"
    return f"{m:.3f} ± {s:.3f}" if row["n_seeds"] > 1 else f"{m:.3f}"


def _md_table(summ, names, first_col):
    head = ["Clean AUC", "Aligned AUC", "Flipped AUC", "Shortcut gap",
            "Worst-group acc", "ΔTPR", "Grad-CAM enrichment"]
    lines = ["| " + first_col + " | " + " | ".join(head) + " |",
             "|" + "---|" * (len(head) + 1)]
    for name, label in names:
        r = summ[summ["name"] == name]
        if r.empty:
            continue
        r = r.iloc[0]
        lines.append("| " + label + " | " + " | ".join(_fmt(r, c) for c in KEY_COLS) + " |")
    return "\n".join(lines)


def make_report(out_dir, rhos, mit_rho, methods, log=print):
    recs = load_runs(out_dir)
    if not recs:
        log("No runs found - nothing to report.")
        return
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    df = pd.DataFrame([_flatten(r) for r in recs])
    df.to_csv(os.path.join(out_dir, "all_runs.csv"), index=False)
    summ = summarise(df)
    summ.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    sweep_names = [("clean_ref", "No shortcut (clean training)")] + [
        (f"erm_rho{r}", f"ρ = {r:.0%}") for r in rhos]
    mit_names = [("clean_ref", METHOD_LABELS["clean_ref"]),
                 (f"erm_rho{mit_rho}", METHOD_LABELS["erm"])] + [
        (f"{m}_rho{mit_rho}", METHOD_LABELS.get(m, m)) for m in methods]

    n_seeds = int(df["seed"].nunique())
    md = [f"# Results\n\nMean ± std over {n_seeds} seed(s). Shortcut gap = AUC(aligned) − "
          "AUC(flipped). ΔTPR = sensitivity on pneumonia images with the marker minus "
          "without it (balanced test set). Grad-CAM enrichment = share of 'pneumonia' heatmap "
          "energy in the marker region ÷ that region's share of the image (1.0 = ignored).\n",
          "## Experiment 1 – shortcut strength sweep (ERM)\n",
          _md_table(summ, sweep_names, "Training data"),
          f"\n## Experiment 2 – mitigations at ρ = {mit_rho:.0%}\n",
          _md_table(summ, mit_names, "Method")]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    _plot_sweep(summ, rhos, fig_dir)
    _plot_mitigation(summ, mit_names, fig_dir)
    _plot_cam_grid(out_dir, sweep_names + mit_names[2:], fig_dir)
    log(f"Report written: {os.path.join(out_dir, 'summary.md')} and {fig_dir}")


def _get(summ, name, col):
    r = summ[summ["name"] == name]
    if r.empty:
        return np.nan, np.nan
    return r.iloc[0][f"{col}_mean"], r.iloc[0][f"{col}_std"]


def _plot_sweep(summ, rhos, fig_dir):
    xs = [r * 100 for r in rhos]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for s, label in (("clean", "Clean test (no marker)"), ("aligned", "Aligned test"),
                     ("flipped", "Flipped test (shortcut reversed)")):
        vals = [_get(summ, f"erm_rho{r}", f"{s}_auc") for r in rhos]
        ax.errorbar(xs, [v[0] for v in vals], yerr=[v[1] for v in vals], marker="o",
                    capsize=3, color=COLORS[s], label=label)
    ref, _ = _get(summ, "clean_ref", "clean_auc")
    if not np.isnan(ref):
        ax.axhline(ref, ls="--", color="grey", lw=1, label="Clean-trained model, clean test")
    ax.axhline(0.5, ls=":", color="black", lw=0.8)
    ax.set_xlabel("Marker–label correlation in training, ρ (%)")
    ax.set_ylabel("Test AUC")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(xs)
    ax.set_title("Performance collapses when the shortcut is reversed")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    labels = ["No\nshortcut"] + [f"ρ={r:.0%}" for r in rhos]
    names = ["clean_ref"] + [f"erm_rho{r}" for r in rhos]
    vals = [_get(summ, n, "mar_enrichment") for n in names]
    ax.bar(labels, [v[0] for v in vals], yerr=[v[1] for v in vals], capsize=3, color="#7a5195")
    ax.axhline(1.0, ls="--", color="grey", lw=1)
    ax.set_ylabel("Grad-CAM enrichment on marker (×)")
    ax.set_title("Grad-CAM attention on the marker")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_shortcut_sweep.png"), dpi=200)
    plt.close(fig)


def _plot_mitigation(summ, mit_names, fig_dir):
    present = [(n, l) for n, l in mit_names if not summ[summ["name"] == n].empty]
    if not present:
        return
    metrics = [("clean_auc", "Clean AUC"), ("flipped_auc", "Flipped AUC"),
               ("worst_group_acc", "Worst-group acc")]
    x = np.arange(len(present))
    w = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    for i, (col, lab) in enumerate(metrics):
        vals = [_get(summ, n, col) for n, _ in present]
        ax.bar(x + (i - 1) * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals],
               capsize=2, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" (", "\n(") for _, l in present], fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.axhline(0.5, ls=":", color="black", lw=0.8)
    ax.set_title("Mitigations: does performance survive a reversed shortcut?")
    ax.legend(fontsize=8)
    ax = axes[1]
    vals = [_get(summ, n, "mar_enrichment") for n, _ in present]
    ax.bar(x, [v[0] for v in vals], yerr=[v[1] for v in vals], capsize=3, color="#7a5195")
    ax.axhline(1.0, ls="--", color="grey", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([l.split(" (")[0] for _, l in present], rotation=30, ha="right",
                       fontsize=8)
    ax.set_title("Grad-CAM enrichment on marker (×)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_mitigations.png"), dpi=200)
    plt.close(fig)


def _plot_cam_grid(out_dir, names, fig_dir):
    rows = []
    for name, label in names:
        f = os.path.join(out_dir, "cams", f"{name}.npz")
        if os.path.exists(f):
            d = np.load(f)
            rows.append((label, d["imgs"], d["cams"], d["labels"]))
    if not rows:
        return
    n_cols = rows[0][1].shape[0]
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(1.9 * n_cols, 1.9 * len(rows)),
                             squeeze=False)
    for r, (label, imgs, cams, labs) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=1)
            ax.imshow(cams[c], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title("normal" if labs[c] == 0 else "pneumonia", fontsize=9)
            if c == 0:
                ax.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle("Grad-CAM for 'pneumonia' on marked test images (marker top-left)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig3_gradcam_grid.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_examples(imgs, labels, size, fig_dir):
    """Figure 0: what the shortcut looks like."""
    from .data import add_marker
    import torch

    os.makedirs(fig_dir, exist_ok=True)
    i_n = int(np.where(labels == 0)[0][0])
    i_p = int(np.where(labels == 1)[0][0])
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.6))
    for ax, (i, marked) in zip(axes, [(i_n, False), (i_n, True), (i_p, False), (i_p, True)]):
        x = torch.from_numpy(imgs[i])[None]
        if marked:
            x = add_marker(x, size)
        ax.imshow(x[0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{'pneumonia' if labels[i] else 'normal'}{' + marker' if marked else ''}",
                     fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig0_examples.png"), dpi=200)
    plt.close(fig)
