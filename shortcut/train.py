"""Training loop, prediction and the three training-time mitigations."""
import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import ShortcutDataset, group_ids
from .model import CXRNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def make_loader(ds, batch_size, train, weights=None, num_workers=0, seed=0, device=None):
    pin = device is not None and device.type == "cuda"
    gen = torch.Generator().manual_seed(seed)
    if train and weights is not None:
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=len(ds), replacement=True, generator=gen)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=pin)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, generator=gen if train else None,
                      num_workers=num_workers, pin_memory=pin)


@torch.no_grad()
def predict(model, ds, device, batch_size=256, num_workers=0):
    """Return (P(pneumonia), labels, marker flags) as numpy arrays."""
    model.eval()
    use_amp = device.type == "cuda"
    probs, ys, ms = [], [], []
    for x, y, m, _ in make_loader(ds, batch_size, False, num_workers=num_workers, device=device):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
        ys.append(y)
        ms.append(torch.as_tensor(m))
    return torch.cat(probs).numpy(), torch.cat(ys).numpy(), torch.cat(ms).numpy()


def train_model(model, train_ds, val_ds, device, epochs, batch_size=64, lr=1e-4,
                weight_decay=1e-4, weights=None, select_best=True, num_workers=0,
                seed=0, log=print, desc=""):
    """Standard fine-tuning. If select_best, keeps the epoch with the best validation AUC.

    Note: the validation set carries the same shortcut as training (as it would in a
    real biased dataset), so validation AUC cannot reveal the shortcut by itself.
    """
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss()
    loader = make_loader(train_ds, batch_size, True, weights, num_workers, seed, device)

    best_auc, best_state, history = -1.0, None, []
    for ep in range(epochs):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y, _, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(y)
            n += len(y)
        sched.step()
        p, yv, _ = predict(model, val_ds, device, num_workers=num_workers)
        val_auc = safe_auc(yv, p)
        history.append({"epoch": ep + 1, "train_loss": total / max(n, 1), "val_auc": val_auc})
        log(f"  {desc} epoch {ep + 1:2d}/{epochs}  loss {total / max(n, 1):.4f}  "
            f"val AUC {val_auc:.4f}  ({time.time() - t0:.1f}s)")
        if select_best and (best_state is None or val_auc > best_auc):
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
    if select_best and best_state is not None:
        model.load_state_dict(best_state)
    recalibrate_bn(model, train_ds, device, batch_size, num_workers)
    return model, history


@torch.no_grad()
def recalibrate_bn(model, ds, device, batch_size=64, num_workers=0):
    """Re-estimate BatchNorm running statistics with one pass over the training data.

    The running averages kept during training lag behind the final weights; recomputing
    them as an exact average makes eval-mode outputs (and the 0.5 decision threshold used
    for accuracy, TPR and worst-group accuracy) match what the network actually learned.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average = exact mean over the pass
    model.train()
    for x, *_ in make_loader(ds, batch_size, True, num_workers=num_workers, device=device):
        model(x.to(device, non_blocking=True))
    for m, mom in zip(bns, momenta):
        m.momentum = mom
    model.eval()


# --------------------------------------------------------------------------
# Mitigations
# --------------------------------------------------------------------------
def group_balanced_weights(labels, markers):
    """Weight each sample by 1 / size of its (label, marker) group, so the sampler draws the
    four groups equally often. Needs the marker labels (i.e. knowing the shortcut)."""
    g = group_ids(labels, markers)
    counts = np.bincount(g, minlength=4).astype(float)
    counts[counts == 0] = np.inf  # empty groups get no weight (e.g. rho = 1.0)
    return 1.0 / counts[g]


def jtt_weights(stage1_model, imgs, labels, markers, size, device, upweight, num_workers=0):
    """Just Train Twice (Liu et al., ICML 2021), stage 2 weights: training samples the briefly
    trained stage-1 model gets wrong are up-weighted. Needs no marker labels."""
    ds = ShortcutDataset(imgs, labels, markers, size, augment=False)
    p, y, _ = predict(stage1_model, ds, device, num_workers=num_workers)
    errors = (p >= 0.5).astype(int) != y
    return np.where(errors, float(upweight), 1.0), errors


def fit(method, data, rho_markers, args, device, seed, log=print, desc=""):
    """Train one model with the given method. Returns (model, history, extra_info)."""
    tr_imgs, tr_labels = data["train"]
    va_imgs, va_labels = data["val"]
    tr_m, va_m = rho_markers
    size = args.size
    common = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                  weight_decay=args.weight_decay, num_workers=args.num_workers,
                  seed=seed, log=log)

    train_ds = ShortcutDataset(tr_imgs, tr_labels, tr_m, size, augment=True,
                               randomize_marker=(method == "marker_random"))
    val_ds = ShortcutDataset(va_imgs, va_labels, va_m, size)
    extra = {}
    weights = None

    if method == "group_balance":
        weights = group_balanced_weights(tr_labels, tr_m)
    elif method == "jtt":
        set_seed(seed)
        stage1 = CXRNet(pretrained=not args.no_pretrained)
        stage1, _ = train_model(stage1, train_ds, val_ds, device, epochs=args.jtt_epochs,
                                batch_size=args.batch_size, lr=args.lr,
                                weight_decay=args.weight_decay, select_best=False,
                                num_workers=args.num_workers, seed=seed, log=log,
                                desc=f"{desc} [JTT stage 1]")
        weights, errors = jtt_weights(stage1, tr_imgs, tr_labels, tr_m, size, device,
                                      args.jtt_upweight, args.num_workers)
        g = group_ids(tr_labels, tr_m)
        extra["jtt_error_set_size"] = int(errors.sum())
        extra["jtt_error_rate_by_group"] = {
            int(k): float(errors[g == k].mean()) if (g == k).any() else None for k in range(4)}
        log(f"  {desc} JTT error set: {errors.sum()} / {len(errors)} training images")
        if errors.sum() == 0:
            log("  WARNING: JTT stage 1 made no training errors; stage 2 equals plain ERM.")
        del stage1
    elif method not in ("erm", "marker_random"):
        raise ValueError(f"unknown method: {method}")

    set_seed(seed)
    model = CXRNet(pretrained=not args.no_pretrained)
    model, history = train_model(model, train_ds, val_ds, device, weights=weights,
                                 desc=desc, **common)
    return model, history, extra
