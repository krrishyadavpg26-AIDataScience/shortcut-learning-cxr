"""Check that Python, PyTorch, the GPU and the dataset are ready.

Usage:  python check_setup.py            (also downloads PneumoniaMNIST 128x128, under 100 MB)
        python check_setup.py --no-data  (skip the download)
"""
import argparse
import importlib
import platform
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-data", action="store_true")
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()
    ok = True

    print(f"Python      : {platform.python_version()} ({sys.executable})")
    if sys.version_info < (3, 10):
        print("  !! Python 3.10+ is required (3.11 recommended)")
        ok = False

    for pkg in ("torch", "torchvision", "medmnist", "sklearn", "pandas", "matplotlib"):
        try:
            mod = importlib.import_module(pkg)
            print(f"{pkg:12s}: {getattr(mod, '__version__', 'installed')}")
        except ImportError:
            print(f"{pkg:12s}: MISSING  -> pip install -r requirements.txt")
            ok = False

    import torch

    print(f"CUDA build  : {torch.version.cuda}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        archs = torch.cuda.get_arch_list()
        print(f"GPU         : {name} | compute capability {cap[0]}.{cap[1]}")
        print(f"Arch list   : {' '.join(archs)}")
        if cap[0] >= 12 and not any(a.startswith("sm_12") for a in archs):
            print("  !! This PyTorch build has no Blackwell (sm_120) kernels.\n"
                  "     Reinstall with: pip install torch torchvision --index-url "
                  "https://download.pytorch.org/whl/cu132")
            ok = False
        try:
            a = torch.randn(2048, 2048, device="cuda")
            (a @ a).sum().item()
            print("GPU test    : matrix multiply OK")
        except Exception as e:  # noqa: BLE001
            print(f"GPU test    : FAILED ({e})")
            ok = False
    else:
        print("GPU         : NOT AVAILABLE - training will run on CPU (slow).\n"
              "  Check: NVIDIA driver installed? PyTorch installed from the cu132 index?")

    if not args.no_data:
        try:
            from shortcut.data import SPLITS, load_split

            for s in SPLITS:
                imgs, labels = load_split(s, 128, args.data_root)
                print(f"Data {s:5s}  : {imgs.shape[0]} images {imgs.shape[1]}x{imgs.shape[2]}, "
                      f"{labels.mean():.1%} pneumonia")
        except Exception as e:  # noqa: BLE001
            print(f"Data        : download FAILED ({e})")
            ok = False

    print("\nAll checks passed." if ok else "\nSome checks failed - see messages above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
