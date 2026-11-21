"""Run the shortcut-learning experiments.

Examples
--------
  python main.py --fake-data --quick --no-pretrained   # 2-minute smoke test, no downloads
  python main.py --quick                               # 1 seed, 3 epochs (~10 min on a GPU)
  python main.py                                       # full run: 3 seeds, 10 epochs
  python main.py --stage report                        # rebuild tables/figures only
  python main.py --only erm_rho0.9 jtt_rho0.9          # run selected configs only
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

from shortcut.data import SPLITS, load_split, train_markers
from shortcut.evaluate import evaluate_model
from shortcut.gradcam import cam_examples, marker_attribution
from shortcut.report import make_report, plot_examples
from shortcut.train import fit, get_device, set_seed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--stage", choices=["all", "train", "report"], default="all")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--size", type=int, default=128, choices=[28, 64, 128, 224])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--rhos", type=float, nargs="+", default=[0.5, 0.7, 0.9, 1.0])
    p.add_argument("--mitigation-rho", type=float, default=0.9)
    p.add_argument("--methods", nargs="+", default=["group_balance", "marker_random", "jtt"],
                   choices=["group_balance", "marker_random", "jtt"])
    p.add_argument("--jtt-epochs", type=int, default=2, help="JTT stage-1 training epochs")
    p.add_argument("--jtt-upweight", type=float, default=20.0, help="JTT lambda_up")
    p.add_argument("--cam-layer", default="layer3", choices=["layer2", "layer3", "layer4"])
    p.add_argument("--only", nargs="+", default=None, help="run only these config names")
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true",
                   help="random init (offline smoke tests only)")
    p.add_argument("--fake-data", action="store_true", help="synthetic images, no download")
    p.add_argument("--quick", action="store_true", help="1 seed, 3 epochs")
    p.add_argument("--overwrite", action="store_true", help="re-run configs already done")
    p.add_argument("--save-models", action="store_true", help="also save model weights")
    args = p.parse_args(argv)
    if args.quick:
        args.epochs = min(args.epochs, 3)
        args.seeds = args.seeds[:1]
    return args


def build_configs(args):
    cfgs = [{"name": "clean_ref", "rho": None, "method": "erm"}]
    cfgs += [{"name": f"erm_rho{r}", "rho": r, "method": "erm"} for r in args.rhos]
    if args.mitigation_rho not in args.rhos:
        cfgs.append({"name": f"erm_rho{args.mitigation_rho}", "rho": args.mitigation_rho,
                     "method": "erm"})
    cfgs += [{"name": f"{m}_rho{args.mitigation_rho}", "rho": args.mitigation_rho, "method": m}
             for m in args.methods]
    if args.only:
        cfgs = [c for c in cfgs if c["name"] in args.only]
    return cfgs


def pick_example_indices(labels, k=3):
    normal = np.where(labels == 0)[0][:k]
    pneu = np.where(labels == 1)[0][:k]
    return np.concatenate([normal, pneu])


def run_one(cfg, seed, data, args, device, log):
    name, rho, method = cfg["name"], cfg["rho"], cfg["method"]
    desc = f"[{name} seed {seed}]"
    t0 = time.time()
    tr_m = train_markers(data["train"][1], "train", rho, seed)
    va_m = train_markers(data["val"][1], "val", rho, seed)
    log(f"\n{desc} method={method} rho={rho} | train marker rate: "
        f"normal {tr_m[data['train'][1] == 0].mean():.2f}, "
        f"pneumonia {tr_m[data['train'][1] == 1].mean():.2f}")

    model, history, extra = fit(method, data, (tr_m, va_m), args, device, seed, log, desc)

    test_imgs, test_labels = data["test"]
    eval_rho = rho if rho is not None else args.mitigation_rho
    metrics = evaluate_model(model, test_imgs, test_labels, args.size, eval_rho, device,
                             args.num_workers)
    cam = marker_attribution(model, test_imgs, args.size, device, layer=args.cam_layer)

    if seed == args.seeds[0]:  # heatmap examples for the figure, first seed only
        idx = pick_example_indices(test_labels)
        imgs, cams = cam_examples(model, test_imgs, idx, args.size, device, args.cam_layer)
        np.savez_compressed(os.path.join(args.out, "cams", f"{name}.npz"), imgs=imgs, cams=cams,
                            labels=test_labels[idx])
    if args.save_models:
        torch.save(model.state_dict(), os.path.join(args.out, "models", f"{name}_seed{seed}.pt"))

    rec = {"name": name, "seed": seed, "rho": rho, "method": method, "metrics": metrics,
           "gradcam": cam, "history": history, "extra": extra,
           "time_sec": round(time.time() - t0, 1),
           "config": {k: getattr(args, k) for k in
                      ("size", "epochs", "batch_size", "lr", "weight_decay", "jtt_epochs",
                       "jtt_upweight", "cam_layer", "fake_data", "no_pretrained")}}
    log(f"{desc} test AUC clean {metrics['clean']['auc']:.3f} | aligned "
        f"{metrics['aligned']['auc']:.3f} | flipped {metrics['flipped']['auc']:.3f} | "
        f"worst-group acc {metrics['balanced']['worst_group_acc']:.3f} | "
        f"Grad-CAM enrichment {cam['enrichment']:.2f}x | {rec['time_sec']}s")
    return rec


def main(argv=None):
    args = parse_args(argv)
    for sub in ("runs", "cams", "figures", "models", "logs"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    log_path = os.path.join(args.out, "logs", f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")
    log_fh = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    device = get_device(args.device)
    log(f"Python {platform.python_version()} | torch {torch.__version__} | device {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type == "cpu" and not args.fake_data:
        log("WARNING: running on CPU. This works but is slow; see the runbook to enable the GPU.")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if args.stage in ("all", "train"):
        log(f"Loading PneumoniaMNIST {args.size}x{args.size}"
            + (" (FAKE synthetic data)" if args.fake_data else "") + " ...")
        data = {s: load_split(s, args.size, args.data_root, args.fake_data) for s in SPLITS}
        for s in SPLITS:
            log(f"  {s:5s}: {len(data[s][1]):5d} images, {data[s][1].mean():.1%} pneumonia")
        plot_examples(*data["test"], args.size, os.path.join(args.out, "figures"))

        cfgs = build_configs(args)
        total = len(cfgs) * len(args.seeds)
        done = 0
        for seed in args.seeds:
            for cfg in cfgs:
                done += 1
                path = os.path.join(args.out, "runs", f"{cfg['name']}_seed{seed}.json")
                if os.path.exists(path) and not args.overwrite:
                    log(f"[{done}/{total}] {cfg['name']} seed {seed}: already done, skipping")
                    continue
                log(f"\n===== [{done}/{total}] {cfg['name']} seed {seed} =====")
                set_seed(seed)
                rec = run_one(cfg, seed, data, args, device, log)
                with open(path, "w") as fh:
                    json.dump(rec, fh, indent=2, default=float)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if args.stage in ("all", "report"):
        make_report(args.out, args.rhos, args.mitigation_rho, args.methods, log)
    log(f"Log saved to {log_path}")
    log_fh.close()


if __name__ == "__main__":  # required on Windows when DataLoader workers > 0
    sys.exit(main())
