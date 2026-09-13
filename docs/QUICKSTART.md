# Quick start — manual setup, notebook workflow

This is the simple path: set the environment up by hand, download the dataset from the
Kaggle website in your browser, and work in Jupyter notebooks inside VS Code.

Everything here assumes Windows + PowerShell. Commands for macOS/Linux are noted where
they differ.

---

## Step 1 — Put the project somewhere sensible

Not in `Downloads` (Windows can auto-clean it) and not in a OneDrive-synced folder
(sync locks cause random file errors during large extractions).

```powershell
mkdir C:\projects -Force
```

Put the `dr_project` folder at `C:\projects\dr_project`, then in VS Code:
`File → Open Folder…` → select `C:\projects\dr_project` (the folder containing `README.md`).

---

## Step 2 — Create the virtual environment

Open a terminal in VS Code (`` Ctrl+` ``). Confirm you are in the right place first:

```powershell
Get-Location          # should print C:\projects\dr_project
dir                   # should list README.md, src, scripts, configs, notebooks
```

Then, one line at a time:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Your prompt should now start with `(.venv)`. If it doesn't, nothing below will work —
see Troubleshooting.

> **If `Activate.ps1` is blocked** with "running scripts is disabled on this system",
> run this once, then try again:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

## Step 3 — Install PyTorch (GPU build) first

Check which CUDA version your driver supports:

```powershell
nvidia-smi
```

Look at "CUDA Version" in the top-right of the output, then install the matching build:

```powershell
# CUDA 12.x driver (most common)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 driver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# No NVIDIA GPU at all
pip install torch torchvision
```

This is ~2.5 GB. Verify before continuing:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

You want `True`. If it says `False`, you installed the CPU wheel — uninstall and redo with
the `--index-url`:

```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 4 — Install everything else

```powershell
pip install -r requirements.txt
```

Optional extras (albumentations, ruff). **Skip this if it errors** — nothing in the project
depends on it:

```powershell
pip install -r requirements-optional.txt
```

---

## Step 5 — Get the dataset (browser download)

1. Open <https://www.kaggle.com/competitions/aptos2019-blindness-detection>
2. Click **Join Competition** and accept the rules (required — even for a manual download)
3. Go to the **Data** tab → **Download All** (~10 GB zip)
4. Unzip it so the layout ends up exactly like this:

```
C:\projects\dr_project\data\raw\aptos2019\
├── train.csv
├── test.csv
├── train_images\        <- 3,662 .png files
└── test_images\
```

**The most common mistake** is one extra nested folder, e.g.
`data\raw\aptos2019\aptos2019-blindness-detection\train.csv`. If that happens, move the
contents up one level. The notebook checks this for you and tells you if it's wrong.

You need roughly 25 GB free while unzipping (zip + extracted copy). Delete the zip afterwards.

> Prefer the scripted download? `python -m scripts.m1_download_data` does the same thing
> with the Kaggle API, once `kaggle.json` is at `%USERPROFILE%\.kaggle\kaggle.json`.

---

## Step 6 — Open the notebook

1. In VS Code: `Ctrl+Shift+P → Python: Select Interpreter` → pick `.venv\Scripts\python.exe`
2. Open `notebooks/01_dataset_inspection.ipynb`
3. Top-right of the notebook → **Select Kernel** → **Python Environments** → `.venv`
4. Run the cells top to bottom with `Shift+Enter`

Each cell prints what it did. Cell 1 will tell you immediately if the project root or the
dataset location is wrong.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `The term '.\.venv\Scripts\python.exe' is not recognized` | Your terminal is not in the project folder. `cd C:\projects\dr_project` first. |
| `No module named 'scripts'` or `No module named 'src'` | Same cause — wrong working directory, or you opened the parent folder in VS Code. |
| Prompt has no `(.venv)` prefix | Run `.\.venv\Scripts\Activate.ps1`, or open a fresh terminal after selecting the interpreter. |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `Microsoft Visual C++ 14.0 or greater is required` | A package with no prebuilt wheel. If it's from `requirements-optional.txt`, just skip that file. |
| `torch.cuda.is_available()` is `False` | CPU-only wheel installed — redo Step 3 with `--index-url`. |
| Notebook kernel won't start | `pip install ipykernel`, then re-pick the kernel. |
| Moved the project and things broke | A venv hard-codes its path. Delete `.venv` and redo Steps 2–4. |
