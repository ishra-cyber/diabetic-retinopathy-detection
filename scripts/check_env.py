"""
MILESTONE 0 - Environment self-check.
File location: <project_root>/scripts/check_env.py

Objective
---------
Before you download 10 GB or start a 3-hour training run, prove that the
environment is actually correct. This script checks, in order:

  1. Python version
  2. That `import src...` resolves (i.e. you are running from the project root)
  3. configs/config.yaml loads
  4. Every required package imports, with its version
  5. PyTorch sees your NVIDIA GPU, and a real tensor operation runs on it
  6. Kaggle credentials are in place
  7. All project directories exist

It prints a PASS/FAIL table and exits with code 0 only if nothing failed.
It also writes reports/environment.txt - paste that into your report's
"Experimental setup" section; examiners ask what hardware you used.

Run
---
    python -m scripts.check_env

In VS Code: F5 -> "M0: Check environment (GPU, packages, Kaggle)"
"""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path

# --- make `src` importable no matter how this script is launched ------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# (import name, friendly name, required?)
REQUIRED_PACKAGES = [
    ("numpy", "NumPy", True),
    ("pandas", "pandas", True),
    ("cv2", "OpenCV", True),
    ("PIL", "Pillow", True),
    ("sklearn", "scikit-learn", True),
    ("matplotlib", "Matplotlib", True),
    ("seaborn", "seaborn", True),
    ("yaml", "PyYAML", True),
    ("tqdm", "tqdm", True),
    ("timm", "timm", True),
    ("albumentations", "albumentations", False),   # optional - see requirements-optional.txt
    ("streamlit", "Streamlit", True),
    ("plotly", "Plotly", True),
    ("reportlab", "ReportLab", True),
    ("kaggle", "Kaggle API", True),
    ("pytest", "pytest", False),
]

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def _supports_colour() -> bool:
    """ANSI colours: only when writing to a real terminal that understands them."""
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    # VS Code's integrated terminal and Windows Terminal handle ANSI natively.
    if os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM") == "vscode":
        return True
    # Older conhost: try to switch on virtual-terminal processing.
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        return bool(kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7))
    except Exception:  # noqa: BLE001
        return False


if not _supports_colour():
    GREEN = RED = YELLOW = RESET = ""

results: list[tuple[str, str, str]] = []   # (status, check name, detail)


def record(ok: bool | None, name: str, detail: str = "") -> None:
    status = "PASS" if ok is True else ("WARN" if ok is None else "FAIL")
    results.append((status, name, detail))
    colour = GREEN if ok is True else (YELLOW if ok is None else RED)
    print(f"  {colour}[{status}]{RESET} {name}" + (f"  -  {detail}" if detail else ""))


# ---------------------------------------------------------------------------
def check_python() -> None:
    v = sys.version_info
    ok = (3, 9) <= (v.major, v.minor) < (3, 13)
    record(ok, "Python version",
           f"{v.major}.{v.minor}.{v.micro} ({sys.executable})"
           + ("" if ok else "  <- need 3.9-3.12; timm/torch wheels lag newer versions"))


def check_project_layout() -> None:
    try:
        from src.utils.config import load_config  # noqa: F401
        record(True, "src package importable", f"root = {PROJECT_ROOT}")
    except ImportError as exc:
        record(False, "src package importable",
               f"{exc}. Open the PROJECT FOLDER in VS Code, not its parent.")


def check_config() -> None:
    try:
        from src.utils.config import load_config
        cfg = load_config()
        record(True, "configs/config.yaml loads",
               f"backbone={cfg['training']['backbone']}, "
               f"image_size={cfg['preprocessing']['image_size']}, "
               f"seed={cfg['project']['seed']}")
    except Exception as exc:  # noqa: BLE001
        record(False, "configs/config.yaml loads", f"{type(exc).__name__}: {exc}")


def check_packages() -> None:
    for module_name, friendly, required in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            record(True, f"import {friendly}", str(version))
        except ImportError:
            hint = "pip install -r requirements.txt"
            # Missing required package -> FAIL; missing optional one -> WARN.
            record(False if required else None, f"import {friendly}",
                   "not installed" + ("" if required else " (optional)") + f" - {hint}")


def check_torch_gpu() -> None:
    try:
        import torch
    except ImportError:
        record(False, "import PyTorch",
               "not installed - install it FIRST with the CUDA index-url, see README")
        return

    record(True, "import PyTorch", f"{torch.__version__} (CUDA build: {torch.version.cuda})")

    # torchvision ships the transforms v2 API this project uses for augmentation.
    try:
        import torchvision
        from torchvision.transforms import v2  # noqa: F401
        record(True, "import torchvision + transforms.v2", torchvision.__version__)
    except ImportError as exc:
        record(False, "import torchvision + transforms.v2",
               f"{exc} - reinstall torch AND torchvision together from the same index-url")

    if not torch.cuda.is_available():
        record(False, "CUDA available to PyTorch",
               "torch.cuda.is_available() is False. Either you installed the CPU-only "
               "wheel (reinstall with --index-url .../cu121) or your NVIDIA driver is "
               "too old. Training will fall back to CPU and take many hours.")
        return

    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    record(True, "CUDA available to PyTorch", f"{name}, {total_gb:.1f} GB VRAM")

    # Actually run something on the GPU - "available" is not the same as "works".
    try:
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        torch.cuda.synchronize()
        result = (a @ b).sum().item()
        record(True, "GPU matmul smoke test", f"ok (checksum {result:.2f})")
    except Exception as exc:  # noqa: BLE001
        record(False, "GPU matmul smoke test", f"{type(exc).__name__}: {exc}")

    # Batch-size guidance based on VRAM, so training does not OOM on the first epoch.
    if total_gb < 6:
        advice = "batch_size 8 at image_size 300, or drop to 256"
    elif total_gb < 10:
        advice = "batch_size 16 at image_size 300 (the config default)"
    elif total_gb < 16:
        advice = "batch_size 24-32 at image_size 300"
    else:
        advice = "batch_size 32-48 at image_size 300"
    record(None, "Suggested training batch size", advice)


def check_nvidia_smi() -> None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0 and out.stdout.strip():
            record(True, "nvidia-smi", out.stdout.strip().splitlines()[0])
        else:
            record(None, "nvidia-smi", "command failed - driver may not be installed")
    except (OSError, subprocess.TimeoutExpired):
        record(None, "nvidia-smi", "not found on PATH (fine if you are not using the GPU yet)")


def check_kaggle_credentials() -> None:
    """Accept any of the four authentication methods the Kaggle CLI supports."""
    home_kaggle = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        record(True, "Kaggle credentials", "KAGGLE_USERNAME / KAGGLE_KEY in environment")
        return
    if os.environ.get("KAGGLE_API_TOKEN"):
        record(True, "Kaggle credentials", "KAGGLE_API_TOKEN in environment (modern token)")
        return

    legacy = home_kaggle / "kaggle.json"
    token = home_kaggle / "access_token"
    if legacy.is_file():
        record(True, "Kaggle credentials", f"{legacy} (legacy key - what this project expects)")
        return
    if token.is_file():
        record(True, "Kaggle credentials", f"{token} (modern API token)")
        return

    record(False, "Kaggle credentials",
           f"none found. Easiest route: kaggle.com/settings > API > 'Create Legacy API Key', "
           f"then move the downloaded kaggle.json to {legacy}")


def check_directories() -> None:
    try:
        from src.utils.config import get_path, load_config
        cfg = load_config()
    except Exception:  # noqa: BLE001
        return  # already reported by check_config
    keys = ["raw_dir", "interim_dir", "processed_dir", "splits_dir",
            "experiments_dir", "reports_dir", "figures_dir"]
    created = []
    for key in keys:
        p = get_path(cfg, key)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            created.append(p.name)
    record(True, "Project directories",
           "all present" if not created else f"created: {', '.join(created)}")


def check_dataset_presence() -> None:
    try:
        from src.utils.config import get_path, load_config
        cfg = load_config()
    except Exception:  # noqa: BLE001
        return
    csv = get_path(cfg, "aptos_dir") / cfg["dataset"]["train_csv"]
    images = get_path(cfg, "aptos_dir") / cfg["dataset"]["train_images"]
    if csv.is_file() and images.is_dir():
        n = sum(1 for _ in images.glob(f"*{cfg['dataset']['image_ext']}"))
        record(True, "APTOS dataset present", f"{n} training images found")
    else:
        record(None, "APTOS dataset present",
               "not downloaded yet - run: python -m scripts.m1_download_data")


def write_environment_report() -> Path:
    """Dump versions + hardware to reports/environment.txt for the report."""
    out = PROJECT_ROOT / "reports" / "environment.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Environment report - generated by scripts/check_env.py",
        "=" * 60,
        f"OS            : {platform.platform()}",
        f"Machine       : {platform.machine()}",
        f"Processor     : {platform.processor()}",
        f"Python        : {sys.version.splitlines()[0]}",
        f"Interpreter   : {sys.executable}",
        "",
    ]
    try:
        import torch
        lines += [
            f"PyTorch       : {torch.__version__} (CUDA build {torch.version.cuda})",
            f"cuDNN         : {torch.backends.cudnn.version()}",
            f"CUDA available: {torch.cuda.is_available()}",
        ]
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            lines += [
                f"GPU           : {props.name}",
                f"VRAM          : {props.total_memory / 1e9:.1f} GB",
                f"Compute cap.  : {props.major}.{props.minor}",
            ]
    except ImportError:
        lines.append("PyTorch       : NOT INSTALLED")

    lines += ["", "Installed packages (pip freeze):", "-" * 60]
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True, timeout=120)
        lines.append(freeze.stdout.strip() or "(pip freeze produced no output)")
    except (OSError, subprocess.TimeoutExpired) as exc:
        lines.append(f"(pip freeze failed: {exc})")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    print("\n" + "=" * 72)
    print("  DR Detection project - environment check")
    print("=" * 72 + "\n")

    print("Interpreter and project layout")
    check_python()
    check_project_layout()
    check_config()

    print("\nPackages")
    check_packages()

    print("\nGPU")
    check_nvidia_smi()
    check_torch_gpu()

    print("\nData access")
    check_kaggle_credentials()
    check_directories()
    check_dataset_presence()

    report = write_environment_report()
    print(f"\nWrote {report.relative_to(PROJECT_ROOT)} - paste this into your report.")

    failures = [r for r in results if r[0] == "FAIL"]
    warnings = [r for r in results if r[0] == "WARN"]

    print("\n" + "=" * 72)
    print(f"  {len(results) - len(failures) - len(warnings)} passed, "
          f"{len(warnings)} warnings, {len(failures)} failed")
    print("=" * 72)

    if failures:
        print("\nFix these before continuing:")
        for _, name, detail in failures:
            print(f"  - {name}: {detail}")
        return 1

    print("\nEnvironment looks good. Next: python -m scripts.m1_download_data\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
