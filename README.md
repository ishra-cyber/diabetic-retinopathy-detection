# Intelligent Diabetic Retinopathy Detection and Progress Monitoring Using AI/ML

An explainable deep-learning web application that grades retinal fundus images into the
five international diabetic retinopathy (DR) severity stages, produces Grad-CAM
heatmaps for every prediction, stores per-patient visit records, and visualises how a
patient's severity changes across visits.

> ## ⚠️ Academic prototype — not a medical device
> This is a final-year student research project built for **screening-support
> demonstration only**. It has not been clinically validated, is not registered with any
> regulatory authority, and must **not** be used for diagnosis, triage, or treatment
> decisions. It does not replace examination by a qualified ophthalmologist.

---

## 1. Severity classes

| Label | Stage |
|------:|-------|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

## 2. Dataset

**APTOS 2019 Blindness Detection** (Kaggle) — 3,662 labelled training fundus images
captured with fundus photography in rural India. Images vary widely in resolution,
illumination, and framing, which is why border cropping and normalisation matter.

The dataset is **not** included in this repository. You download it yourself with your
own Kaggle account (see setup below); Kaggle competition rules must be accepted first.

---

## 3. Setup (local machine with NVIDIA GPU)

### 3.1 Create and activate a virtual environment

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

### 3.2 Install PyTorch with CUDA **first**

Check your driver, then pick the matching wheel index:

```bash
nvidia-smi          # note the "CUDA Version" in the top-right corner
```

```bash
# CUDA 12.1 builds (works with driver >= 530; the usual choice)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 builds (older drivers)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify the GPU is actually visible to PyTorch — do this before you train anything:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### 3.3 Install the rest

```bash
pip install -r requirements.txt
```

### 3.4 Configure Kaggle credentials

1. Open <https://www.kaggle.com/competitions/aptos2019-blindness-detection> and click
   **Join Competition** (accept the rules). *Skipping this causes a 403 error.*
2. Kaggle → Account → **Create New API Token** → downloads `kaggle.json`.

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Windows: place `kaggle.json` at `C:\Users\<you>\.kaggle\kaggle.json`.

---

## 3A. Running the project in VS Code (Windows)

### Step 1 — Open the folder

`File → Open Folder…` → select **`dr_project`** itself (the folder containing
`README.md`). Do **not** open its parent — `import src.utils.config` will fail if the
workspace root is one level too high.

VS Code will prompt *"This workspace has extension recommendations"* → click
**Install All** (Python, Pylance, Jupyter, Ruff, Rainbow CSV, YAML).

### Step 2 — One-command setup

Open a terminal inside VS Code (`Ctrl+` \`) and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

That script creates `.venv`, auto-detects your CUDA version from `nvidia-smi`,
installs the matching PyTorch wheels, installs `requirements.txt`, and runs the
environment self-check. Options:

```powershell
.\setup.ps1 -Cuda cu118      # force an older CUDA build
.\setup.ps1 -Cuda cpu        # no GPU / driver not ready yet
.\setup.ps1 -Recreate        # rebuild .venv from scratch
```

You can also run it from the Command Palette:
`Ctrl+Shift+P → Tasks: Run Task → Setup: create venv + install everything`.

> **If PowerShell refuses to run the script**, that is Windows' execution policy,
> not an error in the file. The `-ExecutionPolicy Bypass` above already handles it;
> if you still get blocked, run the commands in §3.1–3.3 manually.

### Step 3 — Select the interpreter

`Ctrl+Shift+P → Python: Select Interpreter →` choose
`.venv\Scripts\python.exe` (it is labelled *Recommended*).

Check the bottom-right status bar shows `.venv` before running anything. This is the
single most common cause of "but I installed it!" import errors.

### Step 4 — Verify the environment

```powershell
python -m scripts.check_env
```

or press **F5** → **"M0: Check environment (GPU, packages, Kaggle)"**.

You want `CUDA available to PyTorch → PASS` with your GPU name. If it says FAIL with
`torch.cuda.is_available() is False`, you have the CPU-only wheel — re-run
`.\setup.ps1 -Cuda cu121 -Recreate`.

This also writes `reports/environment.txt`, which goes straight into your report's
*Experimental setup* section.

### Step 5 — Run milestones with F5

`launch.json` ships with ready-made configurations:

| F5 configuration | Equivalent command |
|---|---|
| M0: Check environment | `python -m scripts.check_env` |
| M1: Download APTOS dataset | `python -m scripts.m1_download_data` |
| M1: Inspect dataset (sampled) | `python -m scripts.m1_inspect_dataset` |
| M1: Inspect dataset (--scan-all) | `python -m scripts.m1_inspect_dataset --scan-all` |
| Debug: current Python file | runs whatever file is open |
| Streamlit app | `streamlit run app/app.py` *(from Milestone 10)* |

Set a breakpoint by clicking left of a line number, then F5 — you can step through
`src/data/inspect.py` line by line, which is worth doing once so you understand the
integrity checks rather than just trusting them.

### Step 6 — Tests

The Testing beaker icon in the left sidebar discovers `tests/` automatically
(pytest is pre-configured in `.vscode/settings.json`). Or `Ctrl+Shift+P → Tasks: Run
Test Task`, or just:

```powershell
pytest -q
```

### VS Code troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Wrong folder opened, or interpreter not `.venv`. Re-check Steps 1 and 3. |
| Pylance underlines `src.utils.config` in yellow | Reload the window: `Ctrl+Shift+P → Developer: Reload Window`. |
| `torch.cuda.is_available()` is False | CPU-only wheel installed. `.\setup.ps1 -Cuda cu121 -Recreate`. |
| Terminal shows no `(.venv)` prefix | Close the terminal and open a new one after selecting the interpreter. |
| `403 Forbidden` from Kaggle | You have not clicked **Join Competition** on the APTOS page. |
| Anaconda base environment keeps activating | `Ctrl+Shift+P → Python: Select Interpreter` and pick `.venv` explicitly; the workspace setting already points there. |
| Matplotlib window never appears | By design — figures are written to `reports/figures/`, not displayed. |
| `Microsoft Visual C++ 14.0 or greater is required` / `Failed building wheel for stringzilla` | pip found no prebuilt wheel for your Python version and tried to compile C++. That package is `albumentations` → `albucore` → `stringzilla`, which this project does **not** need — it uses `torchvision.transforms.v2`. It now lives in `requirements-optional.txt`; just skip it. If you want it anyway, use Python 3.11 (which has wheels) or install the MS C++ Build Tools. |
| Project is in `Downloads` | Move it to something like `C:\projects\dr_project`. Downloads gets cleaned by Windows Storage Sense, and you are about to put 10 GB of data in there. |


---

## 3B. Prefer notebooks? Start here

If you would rather work interactively than run scripts, see **`docs/QUICKSTART.md`** —
manual venv setup, browser dataset download, and the notebook workflow, step by step.

Milestone notebooks live in `notebooks/` and import their logic from `src/`, so the same
code that produces your report figures is the code the Streamlit app uses later:

| Notebook | Milestone |
|---|---|
| `01_dataset_inspection.ipynb` | 1 — acquisition and inspection |
| `02_preprocessing.ipynb` | 2 — preprocessing and augmentation *(next)* |

---

## 4. Running Milestone 1

```bash
# from the project root
python -m scripts.m1_download_data        # downloads + extracts APTOS 2019
python -m scripts.m1_inspect_dataset      # integrity checks, stats, figures
```

Outputs land in `reports/` and `reports/figures/`.

Run the unit tests at any time:

```bash
pytest -q
```

---

## 5. Folder structure

```
dr_project/
├── configs/config.yaml          # single source of truth for paths + hyper-parameters
├── data/
│   ├── raw/aptos2019/           # Kaggle download (git-ignored)
│   ├── interim/                 # cached preprocessed images
│   └── splits/                  # train/val/test CSVs (seeded, committed)
├── src/
│   ├── data/                    # inspection, preprocessing, dataset, splits
│   ├── models/                  # backbone factory, train loop, evaluation
│   ├── explain/                 # Grad-CAM
│   ├── db/                      # SQLite schema + data-access layer
│   ├── analysis/                # progression logic + charts
│   └── utils/                   # config, seeding, logging
├── scripts/                     # one runnable entry point per milestone step
├── app/                         # Streamlit application + assets
├── experiments/                 # checkpoints + metrics, one folder per run
├── reports/                     # figures, tables, screenshots, report text
├── tests/                       # pytest suite
├── docs/                        # architecture notes, report sections
├── .vscode/                     # launch.json, tasks.json, settings.json
├── setup.ps1 / setup.sh         # one-command environment setup
└── pyproject.toml               # pytest + ruff configuration
```

---

## 6. Milestones

| # | Milestone | Status |
|--:|-----------|--------|
| 1 | Dataset acquisition and inspection | ✅ |
| 2 | Preprocessing and augmentation | ⬜ |
| 3 | Stratified train/val/test split | ⬜ |
| 4 | Baseline transfer-learning model | ⬜ |
| 5 | Imbalance handling and training | ⬜ |
| 6 | Evaluation and experiment comparison | ⬜ |
| 7 | Grad-CAM explainability | ⬜ |
| 8 | SQLite patient-visit database | ⬜ |
| 9 | Progression analysis and charts | ⬜ |
| 10 | Streamlit web application | ⬜ |
| 11 | Testing, documentation, presentation | ⬜ |

## 7. Reproducibility

Every script calls `set_seed(cfg["project"]["seed"])` before doing anything random.
For the final run you report in your paper, use `set_seed(seed, deterministic=True)`
and record the exact commit hash, GPU model, and library versions
(`pip freeze > reports/environment.txt`).

## 8. References

See `docs/references.md` (populated in Milestone 11).
