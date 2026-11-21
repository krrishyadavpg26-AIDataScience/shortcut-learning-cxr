"""Grad-CAM (Selvaraju et al., ICCV 2017) and the marker attribution score."""
import numpy as np
import torch
import torch.nn.functional as F

from .data import add_marker, attribution_box


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.acts = None
        self.grads = None
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.acts = output
        if output.requires_grad:
            output.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self.grads = grad

    def remove(self):
        self.handle.remove()

    def __call__(self, x, target_class=1):
        """Heatmaps (N, H, W) for target_class, upsampled to the input size."""
        self.model.eval()
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            logits[:, target_class].sum().backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)          # alpha_k
        cam = torch.relu((weights * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return cam[:, 0].detach().float()


def marker_attribution(model, imgs, size, device, layer="layer3", batch_size=64,
                       target_class=1):
    """Put the marker on every test image and measure where Grad-CAM for 'pneumonia' looks.

    Returns:
      mar        - mean share of heatmap energy inside the marker region (0..1)
      area_frac  - that region's share of the image (what a model ignoring it would score)
      enrichment - mar / area_frac (1.0 = no special attention; higher = shortcut use)
    """
    y0, y1, x0, x1 = attribution_box(size)
    area_frac = (y1 - y0) * (x1 - x0) / float(size * size)
    cam_fn = GradCAM(model, model.cam_layer(layer))
    shares = []
    try:
        for i in range(0, len(imgs), batch_size):
            x = torch.from_numpy(imgs[i:i + batch_size]).unsqueeze(1).to(device)
            x = add_marker(x, size)
            cam = cam_fn(x, target_class)
            total = cam.sum(dim=(1, 2))
            inside = cam[:, y0:y1, x0:x1].sum(dim=(1, 2))
            ok = total > 0
            shares.extend((inside[ok] / total[ok]).cpu().tolist())
    finally:
        cam_fn.remove()
    mar = float(np.mean(shares)) if shares else float("nan")
    return {"mar": mar, "area_frac": area_frac, "enrichment": mar / area_frac,
            "n_scored": len(shares)}


def cam_examples(model, imgs, idx, size, device, layer="layer3", target_class=1):
    """Marked example images and their normalised heatmaps, for the figure grid."""
    x = torch.from_numpy(imgs[idx]).unsqueeze(1).to(device)
    x = add_marker(x, size)
    cam_fn = GradCAM(model, model.cam_layer(layer))
    try:
        cam = cam_fn(x, target_class)
    finally:
        cam_fn.remove()
    cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return x[:, 0].cpu().numpy(), cam.cpu().numpy()
