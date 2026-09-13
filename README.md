# Intelligent Diabetic Retinopathy Detection and Progress Monitoring Using AI/ML

An explainable deep-learning system that grades retinal fundus photographs into the five
international diabetic retinopathy severity stages, produces a Grad-CAM heatmap for every
prediction, stores per-patient visit records, and visualises how predicted severity changes
across visits.

Three CNN architectures are trained under identical conditions and compared with bootstrap
confidence intervals.

---

> ## ⚠️ Academic prototype — not a medical device
>
> This is a final-year student research project built for **screening-support demonstration
> only**. It has not been clinically validated, is not registered with any regulatory
> authority, and must **not** be used for diagnosis, triage, or treatment decisions. It does
> not replace examination by a qualified ophthalmologist.
>
> Multi-visit patient histories in this repository are **simulated**. See
> [Limitations](#limitations).

---

## Contents

- [Results](#results)
- [What the system does](#what-the-system-does)
- [Dataset](#dataset)
- [Method](#method)
- [Reproducing this work](#reproducing-this-work)
- [Repository structure](#repository-structure)
- [What is deliberately not in this repository](#what-is-deliberately-not-in-this-repository)
- [Limitations](#limitations)
- [References](#references)

---

## Results

Held-out test split (n = 550 images), evaluated **once** after all model selection was
complete. Confidence intervals are 95% percentile bootstrap over 1,000 resamples.

| Architecture | Parameters | Test QWK | 95% CI |
|---|---:|---:|---|
| **DenseNet121** | 6.96 M | **0.8910** | [0.865, 0.916] |
| ResNet50 | 23.52 M | 0.8773 | [0.847, 0.904] |
| EfficientNet-B3 | 10.70 M | 0.8583 | [0.824, 0.888] |

Validation QWK during model selection: DenseNet121 0.8992, ResNet50 0.8949,
EfficientNet-B3 0.8938.

**Quadratic Weighted Kappa (QWK) is the primary metric**, not accuracy. Diabetic retinopathy
grading is *ordinal*: predicting stage 4 when the truth is stage 0 is a far worse error than
predicting stage 1. QWK penalises errors by the square of their distance. It is also the
official APTOS 2019 competition metric, making these numbers comparable to published work.

The argument in one sentence: *a degenerate classifier predicting "No DR" for every image
scores roughly 49% accuracy on this dataset and a QWK of 0.*

### On the architecture comparison

All three models were trained on identical data (same stratified split, verified by a shared
split fingerprint), identical preprocessing, identical augmentation, identical weighted
cross-entropy loss and identical optimisation schedule. The only variable was the
architecture.

The spread across architectures is smaller than the width of a single confidence interval.
Paired bootstrap comparisons should be consulted before claiming any architecture is
superior — see `reports/m6_test_pairwise.csv`.

Worth noting: **DenseNet121 achieved the highest QWK with under a third of ResNet50's
parameters**, which suggests that on a dataset of this size the limiting factor is training
data volume rather than model capacity.

---

## What the system does

```
┌──────────────────────── PRESENTATION ────────────────────────┐
│  Streamlit app                                               │
│   Screen · Patient history · Database · About                │
└──────────▲──────────────────────────────────▲────────────────┘
     inference request                  history queries
           │                                  │
┌──────────┴──────────── APPLICATION ─────────┴────────────────┐
│  Inference + Grad-CAM         Progression analysis            │
│  Data-access layer ──────── SQLite (patients, visits)         │
└──────────▲───────────────────────────────────────────────────┘
           │
┌──────────┴─────────────── MODEL ─────────────────────────────┐
│  APTOS 2019 → preprocess → stratified split → CNN            │
│  → weighted cross-entropy → checkpoint + metrics             │
└──────────────────────────────────────────────────────────────┘
```

| Capability | Detail |
|---|---|
| Severity grading | 5 classes: No DR, Mild, Moderate, Severe, Proliferative DR |
| Explainability | Grad-CAM on the final convolutional layer, with border-attention diagnostics |
| Visit records | SQLite: patient ID, date, predicted stage, confidence, full probability vector, image and heatmap paths, model version |
| Progression | Trend classification (improving / stable / worsening), least-squares slope in stages per year, step chart |
| Reporting | Downloadable per-patient PDF, disclaimer first |

---

## Dataset

**APTOS 2019 Blindness Detection** — 3,662 labelled fundus photographs captured in rural
India across several camera models.

| Stage | Class | Images | Share |
|---:|---|---:|---:|
| 0 | No DR | 1,805 | 49.3% |
| 1 | Mild | 370 | 10.1% |
| 2 | Moderate | 999 | 27.3% |
| 3 | Severe | 193 | 5.3% |
| 4 | Proliferative DR | 295 | 8.1% |

Class imbalance ratio **9.4 : 1**. Image dimensions span 640×480 to 4288×2848.

The dataset is **not** included here — see
[What is deliberately not in this repository](#what-is-deliberately-not-in-this-repository).

---

## Method

### Preprocessing

```
read → crop black borders → pad to square → [optional CLAHE] → resize 300×300
```

Cropping removes on average 7.7% of pixels (mean black area falls from 21.4% to 15.4%).
Its value is **not** bulk pixel reduction but *normalisation*: the amount of black surround
varies between images, so resizing alone would leave the retina occupying a different
fraction of each input, presenting identical lesions at different scales.

Padding to square before resizing preserves aspect ratio, so a circular optic disc does not
become an ellipse.

Preprocessed images are cached once to `data/interim/train_300/`. The same
`preprocess_from_config()` runs in training and in the app, so inference cannot drift from
what evaluation measured.

### Augmentation

Geometry is generous, colour is deliberately conservative:

| | |
|---|---|
| Horizontal / vertical flip | p = 0.5 each |
| Rotation | ±15° |
| Scale | 0.90–1.00 |
| Brightness / contrast / saturation | 0.15 / 0.15 / 0.10 |
| **Hue** | **0.0 — unchanged** |

A fundus photograph has no canonical orientation (cameras rotate; left and right eyes are
mirror images), so flips and rotation are free label-preserving variety. Colour is different:
haemorrhages are dark red and exudates pale yellow, so lesion identity is partly *in* the
colour. Shifting hue would corrupt the label while leaving the image looking plausible.

### Split

Stratified 70 / 15 / 15 (2,562 / 550 / 550), seed 42, written to `data/splits/` and
committed. Every run reads those CSVs and records a **split fingerprint**, so two runs can be
proven to have trained on identical data before their scores are compared.

### Training

| | |
|---|---|
| Transfer learning | ImageNet-pretrained backbones via `timm`, head replaced with `Linear(→5)` |
| Loss | Weighted cross-entropy, weights from inverse class frequency **on the train split only** |
| Optimiser | AdamW, cosine schedule with 1 epoch linear warmup |
| Epochs | 20, early stopping on validation QWK (patience 5) |
| Precision | Mixed (AMP) |
| Effective batch | 16 (gradient accumulation) |
| Hardware | NVIDIA RTX 3050 6 GB Laptop GPU, CUDA 12.1 |

Class weights are computed from the training split alone; using whole-dataset counts would
leak test-set label statistics into training.

---

## Reproducing this work

### 1. Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# PyTorch first, with the CUDA build matching your driver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify:

```powershell
python -m scripts.check_env
```

Tested on Python 3.11.9, torch 2.5.1+cu121, Windows 11.

### 2. Dataset

Accept the rules at
[kaggle.com/competitions/aptos2019-blindness-detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection),
then either place `kaggle.json` at `~/.kaggle/kaggle.json` and run the download script, or
download manually. Arrange as:

```
data/raw/
├── train.csv
└── train_images/     3,662 .png files
```

### 3. Pipeline

```powershell
python -m scripts.m1_inspect_dataset            # integrity checks, class distribution
python -m scripts.m2_cache_images --workers 4   # cache 300×300 images
python -m scripts.m3_split                      # stratified split (run once)

python -m scripts.m4_train --experiment resnet50
python -m scripts.m4_train --experiment efficientnet_b3
python -m scripts.m4_train --experiment densenet121

python -m scripts.m6_evaluate                   # test metrics + figures + bootstrap CIs
python -m scripts.m7_gradcam                    # explainability figures

python -m scripts.m8_init_db --demo             # database + simulated histories
python -m scripts.m9_progression                # progression charts
streamlit run app/app.py                        # the application
```

Smoke-test before any long run:

```powershell
python -m scripts.m4_train --experiment densenet121 --smoke
```

### 4. Tests

```powershell
pytest -q
```

73 tests covering dataset integrity checks, the preprocessing pipeline, split correctness and
leakage detection, metrics, the database layer, and progression logic.

### Notebooks

| Notebook | Milestones |
|---|---|
| `01_dataset_inspection.ipynb` | Dataset acquisition and exploratory analysis |
| `02_preprocessing.ipynb` | Preprocessing and augmentation |
| `03_split_and_training.ipynb` | Split, model pre-flight, training curves |
| `04_evaluation_and_gradcam.ipynb` | Test evaluation, bootstrap CIs, Grad-CAM |

Notebooks import from `src/` rather than defining logic inline, so the code producing the
report figures is the same code the application runs.

---

## Repository structure

```
├── app/app.py                  Streamlit application
├── configs/
│   ├── config.yaml             all paths and hyper-parameters
│   └── experiments/            one override file per architecture
├── data/splits/                the exact train/val/test partition (committed)
├── docs/                       architecture notes, quickstart, references
├── experiments/<run>/          per run: metrics, history, config snapshot, logs
├── notebooks/                  four milestone notebooks
├── reports/
│   ├── figures/                every figure, all generated
│   └── *.csv, *.json           every results table
├── scripts/                    one entry point per milestone step
├── src/
│   ├── data/                   inspection, preprocessing, transforms, dataset, split
│   ├── models/                 factory, training, metrics, evaluation, bootstrap, figures
│   ├── explain/                Grad-CAM
│   ├── db/                     SQLite schema and data-access layer
│   ├── analysis/               progression analysis
│   └── utils/                  config, seeding, logging
└── tests/                      pytest suite
```

---

## What is deliberately not in this repository

| | Why | How to obtain |
|---|---|---|
| `data/raw/` — the APTOS images | ~10 GB, and Kaggle's competition rules do not permit redistribution | Download from Kaggle after accepting the rules |
| `.venv/` | Environment, not source | `pip install -r requirements.txt` |
| `*.pt` model checkpoints | 30–90 MB each; bloats every clone | Attached to the GitHub **Releases** page, or retrain with the commands above |
| `app/assets/*.sqlite3` | Runtime state, regenerable | `python -m scripts.m8_init_db --demo` |
| `kaggle.json` | Credential | Yours, from your Kaggle account |

Everything needed to reproduce the results — code, configs, the exact data split, and all
generated metrics — **is** committed.

---

## Limitations

Stated plainly, because a screening tool that overstates itself is worse than no tool.

- **Single dataset, no external validation.** Trained and evaluated entirely on APTOS 2019.
  Performance on images from different cameras, populations, or capture protocols is unknown.
- **Small rare classes.** Severe DR is 5.3% of the data — roughly 29 images in the test
  split. Per-class metrics for that stage carry wide uncertainty.
- **No clinician review.** No ophthalmologist validated any prediction or heatmap in this
  project.
- **Simulated longitudinal data.** APTOS is cross-sectional: one image per patient, no
  follow-up. The progression module is demonstrated on **synthetic** visit histories. Every
  simulated record is flagged `is_simulated=1` in the database, marked in the UI and on every
  chart, and labelled in the PDF report. The progression *logic* is real and would run
  unchanged on real longitudinal data.
- **Grad-CAM localises regions, not lesions.** The heatmap is roughly 10×10 upsampled to
  300×300. A plausible-looking heatmap is not proof the model reasoned correctly.
- **Architectures are not clearly separated.** Differences between the three models are small
  relative to their confidence intervals.
- **Suggested review intervals are illustrative only** and are not clinical guidance.
- **Not deployed publicly, deliberately.** A publicly reachable tool that returns a severity
  grade for an uploaded retinal photograph carries a real risk of misuse regardless of any
  disclaimer.

## Future work

The architecture comparison suggests capacity is not the bottleneck, which points the next
steps at data rather than models:

- External validation on EyePACS (Kaggle DR 2015, ~35,000 images)
- Pretraining on EyePACS before fine-tuning on APTOS
- Ordinal regression loss instead of categorical cross-entropy, matching the metric
- Test-time augmentation and ensembling
- Higher input resolution for microaneurysm detection, given sufficient GPU memory
- Lesion-level annotation to validate Grad-CAM against ground-truth pathology locations

---

## References

1. Asia Pacific Tele-Ophthalmology Society, "APTOS 2019 Blindness Detection," Kaggle, 2019.
2. C. P. Wilkinson et al., "Proposed international clinical diabetic retinopathy and diabetic
   macular edema disease severity scales," *Ophthalmology*, vol. 110, no. 9, pp. 1677–1682, 2003.
3. V. Gulshan et al., "Development and validation of a deep learning algorithm for detection
   of diabetic retinopathy in retinal fundus photographs," *JAMA*, vol. 316, no. 22,
   pp. 2402–2410, 2016.
4. G. Huang, Z. Liu, L. van der Maaten, K. Q. Weinberger, "Densely connected convolutional
   networks," in *Proc. IEEE CVPR*, 2017, pp. 4700–4708.
5. K. He, X. Zhang, S. Ren, J. Sun, "Deep residual learning for image recognition," in
   *Proc. IEEE CVPR*, 2016, pp. 770–778.
6. M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural
   networks," in *Proc. ICML*, 2019, pp. 6105–6114.
7. R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via
   gradient-based localization," in *Proc. IEEE ICCV*, 2017, pp. 618–626.
8. J. Cohen, "Weighted kappa: Nominal scale agreement with provision for scaled disagreement
   or partial credit," *Psychological Bulletin*, vol. 70, no. 4, pp. 213–220, 1968.
9. B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. Chapman & Hall, 1993.
10. R. Wightman, "PyTorch Image Models (timm)," GitHub, 2019.

---

## Licence and use

Academic coursework. The APTOS 2019 dataset remains subject to Kaggle's competition rules.
This software is not licensed, certified, or approved for any clinical purpose.
