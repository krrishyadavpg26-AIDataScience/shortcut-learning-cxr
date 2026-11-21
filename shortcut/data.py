"""Data loading and synthetic shortcut injection.

A "shortcut" here is a small white square (the marker) placed in the top-left
corner of an X-ray. During training we correlate the marker with the label:

    P(marker | pneumonia) = rho        P(marker | normal) = 1 - rho

rho = 0.5 means the marker carries no information; rho = 1.0 means the marker
alone perfectly predicts the label.
"""
import os

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

LABEL_NAMES = {0: "normal", 1: "pneumonia"}
SPLITS = ("train", "val", "test")

# Fixed seeds so every model is tested on exactly the same marked test images.
_TEST_MARKER_SEEDS = {"none": 0, "correlated": 101, "flipped": 202, "random": 303}


# --------------------------------------------------------------------------
# Marker geometry
# --------------------------------------------------------------------------
def marker_box(size):
    """(y0, y1, x0, x1) of the marker square: 10x10 px at offset 4 px for 128x128."""
    side = max(4, round(size * 10 / 128))
    off = max(2, round(size * 4 / 128))
    return off, off + side, off, off + side


def attribution_box(size):
    """Region used to score Grad-CAM attention: the marker plus a one-marker-wide margin."""
    y0, y1, x0, x1 = marker_box(size)
    pad = y1 - y0
    return max(0, y0 - pad), min(size, y1 + pad), max(0, x0 - pad), min(size, x1 + pad)


def add_marker(x, size):
    """Paint the marker onto a (C, H, W) or (N, C, H, W) tensor with values in [0, 1]."""
    y0, y1, x0, x1 = marker_box(size)
    x = x.clone()
    x[..., y0:y1, x0:x1] = 1.0
    return x


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_split(split, size=128, root="data", fake=False):
    """Return (images float32 in [0,1] with shape (N, H, W), labels int64 with shape (N,))."""
    if fake:
        return make_fake_split(split, size)
    from medmnist import PneumoniaMNIST  # imported lazily so --fake-data works without it

    os.makedirs(root, exist_ok=True)
    ds = PneumoniaMNIST(split=split, size=size, root=root, download=True)
    imgs = np.asarray(ds.imgs)
    if imgs.ndim == 4:  # (N, H, W, C) -> keep one channel
        imgs = imgs[..., 0]
    imgs = imgs.astype(np.float32) / 255.0
    labels = np.asarray(ds.labels).reshape(-1).astype(np.int64)
    return imgs, labels


def make_fake_split(split, size=128):
    """Small synthetic dataset for smoke-testing the pipeline without downloading anything.

    Pneumonia images get a soft bright blob, so there is a real (if easy) signal to learn.
    """
    n = {"train": 600, "val": 150, "test": 200}[split]
    rng = np.random.default_rng(424242 + {"train": 0, "val": 1, "test": 2}[split])
    labels = (rng.random(n) < 0.7).astype(np.int64)
    yy, xx = np.mgrid[0:size, 0:size] / size
    base = 0.35 + 0.10 * np.sin(6 * xx)
    imgs = np.empty((n, size, size), np.float32)
    for i in range(n):
        img = base + 0.08 * rng.standard_normal((size, size))
        if labels[i] == 1:
            cy, cx = rng.uniform(0.4, 0.7), rng.uniform(0.25, 0.75)
            img = img + 0.25 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 0.08 ** 2))
        imgs[i] = np.clip(img, 0.0, 0.95)
    return imgs, labels


# --------------------------------------------------------------------------
# Marker assignment
# --------------------------------------------------------------------------
def assign_markers(labels, mode, rho=0.5, rng=None):
    """Decide which images carry the marker.

    mode:
      "none"       - no image has the marker
      "correlated" - P(marker|pneumonia)=rho, P(marker|normal)=1-rho
      "flipped"    - marker on every normal image and no pneumonia image (shortcut reversed)
      "random"     - marker on a random 50%, independent of the label
      "all"        - every image has the marker
    """
    labels = np.asarray(labels)
    n = len(labels)
    rng = rng if rng is not None else np.random.default_rng(0)
    if mode == "none":
        return np.zeros(n, dtype=bool)
    if mode == "all":
        return np.ones(n, dtype=bool)
    if mode == "flipped":
        return labels == 0
    if mode == "random":
        p = np.full(n, 0.5)
    elif mode == "correlated":
        p = np.where(labels == 1, rho, 1.0 - rho)
    else:
        raise ValueError(f"unknown marker mode: {mode}")
    return rng.random(n) < p


def train_markers(labels, split, rho, seed):
    """Markers for train/val. The same uniform draws are reused for every rho, so a larger
    rho only ever adds markers to pneumonia images and removes them from normal ones."""
    if rho is None:
        return np.zeros(len(labels), dtype=bool)
    rng = np.random.default_rng(10007 + seed * 7919 + {"train": 0, "val": 1}[split])
    return assign_markers(labels, "correlated", rho, rng)


def test_markers(labels, mode, rho=0.5):
    """Markers for a test set; seeded per mode so all models see identical test images."""
    rng = np.random.default_rng(_TEST_MARKER_SEEDS.get(mode, 999) + int(round(rho * 100)))
    return assign_markers(labels, mode, rho, rng)


def group_ids(labels, markers):
    """Group index = 2*label + marker -> 0 normal/no-marker, 1 normal/marker,
    2 pneumonia/no-marker, 3 pneumonia/marker."""
    return 2 * np.asarray(labels).astype(int) + np.asarray(markers).astype(int)


GROUP_NAMES = {0: "normal_nomarker", 1: "normal_marker", 2: "pneumonia_nomarker", 3: "pneumonia_marker"}


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class ShortcutDataset(Dataset):
    """Returns (image (1,H,W), label, marker_flag, index)."""

    def __init__(self, imgs, labels, markers, size, augment=False, randomize_marker=False):
        self.imgs = torch.from_numpy(np.ascontiguousarray(imgs)).unsqueeze(1)
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.markers = torch.from_numpy(np.asarray(markers, dtype=np.int64))
        self.size = size
        self.randomize_marker = randomize_marker
        # Mild geometric augmentation. No horizontal flip: it would mirror chest anatomy.
        self.tf = (T.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05))
                   if augment else None)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.imgs[i]
        if self.tf is not None:
            x = self.tf(x)
        m = int(self.markers[i])
        if self.randomize_marker:  # mitigation: marker becomes independent of the label
            m = int(torch.rand(()) < 0.5)
        if m:
            x = add_marker(x, self.size)  # added after augmentation so it never moves
        return x, self.labels[i], m, i
