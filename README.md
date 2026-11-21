# Shortcut Learning in Chest X-ray Classification

**Can a pneumonia classifier be fooled by a tiny marker, can Grad-CAM catch it, and which fix works?**

Machine Learning course project (Bhatia Sir), MTech AI & Data Science, NSUT.

Deep chest X-ray models often learn *shortcuts* — hospital tags, chest drains, ECG leads — instead
of lung disease. In real data the shortcut is hidden, so it can't be measured. This project plants
a known shortcut (a 10×10 white square, top-left) in **PneumoniaMNIST** and:

1. **Measures reliance** as the marker–label correlation ρ goes 50% → 70% → 90% → 100%.
2. **Tests detection**: does Grad-CAM put attention on the marker in proportion to that reliance?
3. **Compares fixes** at ρ = 90%: group-balanced sampling, marker randomisation, Just Train Twice (JTT).

Key papers: Sourget et al., *Mask of Truth*, J. Imaging Inform. Med. 2025
([arXiv:2412.04030](https://arxiv.org/abs/2412.04030)); Boland, Tsaftaris & Dahdouh, MELBA 2025
([arXiv:2511.17421](https://arxiv.org/abs/2511.17421)).

## Quick start (Windows 11, NVIDIA RTX 50-series)

```powershell
git clone https://github.com/krrishyadavpg26-AIDataScience/shortcut-learning-cxr.git
cd shortcut-learning-cxr
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt

python check_setup.py                                 # GPU + dataset check
python main.py --fake-data --quick --no-pretrained    # 2-min pipeline smoke test
python main.py --quick --out results_quick            # ~10 min real trial run
python main.py                                        # full experiment (~30-45 min)
```

On Linux/macOS use `python3.11 -m venv .venv` and `source .venv/bin/activate`.
RTX 50-series (Blackwell, sm_120) GPUs need a PyTorch build with CUDA 12.8 or newer — that is why
PyTorch is installed from the `cu132` index. Other NVIDIA GPUs work with the same command.

## What gets trained

| Config name | Training data | Method |
|---|---|---|
| `clean_ref` | no markers | ERM |
| `erm_rho0.5` … `erm_rho1.0` | marker on ρ of pneumonia, 1−ρ of normal | ERM |
| `group_balance_rho0.9` | ρ = 0.9 | sampler equalises the 4 (label × marker) groups |
| `marker_random_rho0.9` | ρ = 0.9 | marker re-drawn at random each time an image is seen |
| `jtt_rho0.9` | ρ = 0.9 | 2-epoch ERM → up-weight its errors ×20 → retrain |

8 configs × 3 seeds = 24 runs. Each run: ImageNet ResNet-18, 128×128, AdamW lr 1e-4,
batch 64, 10 epochs, cosine schedule, best epoch by validation AUC, then BatchNorm statistics
re-estimated with one pass over the training set (keeps the 0.5 decision threshold reliable).

Every model is tested on four versions of the same 624 test images:

| Test set | Markers | Tells you |
|---|---|---|
| clean | none | true disease performance |
| aligned | same correlation as training | the optimistic, "benchmark" number |
| flipped | only on normal images | what happens when the shortcut misleads |
| balanced | random 50% | worst-group accuracy and ΔTPR |

## Outputs (`results/`)

| File | Contents |
|---|---|
| `summary.md` | the two results tables (mean ± std over seeds) — paste into slides |
| `summary.csv`, `all_runs.csv` | same numbers, machine-readable |
| `figures/fig0_examples.png` | what the marker looks like |
| `figures/fig1_shortcut_sweep.png` | AUC vs ρ on each test set + Grad-CAM enrichment |
| `figures/fig2_mitigations.png` | the four methods compared |
| `figures/fig3_gradcam_grid.png` | heatmaps for every model on the same images |
| `runs/*.json` | full metrics, training curves and JTT stats per run |
| `logs/*.log` | console log of each session |

Runs are resumable: re-running `python main.py` skips any config whose JSON already exists.
Use `--overwrite` to redo them, or `--stage report` to only rebuild tables and figures.

## Metrics

- **AUC** on each test set.
- **Shortcut gap** = AUC(aligned) − AUC(flipped). Near 0 = no shortcut use.
- **Worst-group accuracy**: lowest accuracy over the four (label × marker) groups, balanced set.
- **ΔTPR**: sensitivity on pneumonia images with the marker minus without it (Boland et al. 2025).
- **Grad-CAM enrichment**: share of the "pneumonia" Grad-CAM energy (ResNet `layer3`) falling in
  the marker region, divided by that region's share of the image. 1.0 = the model ignores it.

## Repository layout

```
main.py              experiment runner (train → evaluate → Grad-CAM → report)
check_setup.py       environment and dataset check
shortcut/data.py     loading, marker injection, datasets
shortcut/model.py    ResNet-18 wrapper for grayscale input
shortcut/train.py    training loop, group balancing, JTT
shortcut/evaluate.py test sets and metrics
shortcut/gradcam.py  Grad-CAM and marker attribution
shortcut/report.py   tables and figures
docs/GENAI_USAGE.md  log of generative-AI use (course requirement)
```

## Useful options

```
--epochs 10  --seeds 0 1 2  --rhos 0.5 0.7 0.9 1.0  --mitigation-rho 0.9
--methods group_balance marker_random jtt  --jtt-epochs 2  --jtt-upweight 20
--cam-layer layer3  --size 128  --batch-size 64  --only <config names>  --save-models
```

## Data and licence

PneumoniaMNIST (Yang et al., *Scientific Data* 2023; source images Kermany et al., *Cell* 2018) is
downloaded automatically by the `medmnist` package into `data/`. Code: MIT licence.
