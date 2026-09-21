# Intelligent Diabetic Retinopathy Detection and Progress Monitoring Using AI/ML

An explainable deep-learning system that grades retinal fundus photographs into the five
international diabetic retinopathy severity stages, produces a Grad-CAM heatmap for every
prediction, stores per-patient visit records, and visualises how predicted severity changes
across visits.

Seven models are trained under controlled conditions and compared with paired bootstrap
confidence intervals — three architectures, and a four-arm ablation isolating the effect of
the loss function and the augmentation regime.

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
- [Beyond the baseline](#beyond-the-baseline)
- [Reproducing this work](#reproducing-this-work)
- [Repository structure](#repository-structure)
- [What is deliberately not in this repository](#what-is-deliberately-not-in-this-repository)
- [Limitations](#limitations)
- [Future work](#future-work)
- [References](#references)

---

## Results

Held-out test split (n = 550 images), evaluated after all model selection was complete.
Confidence intervals are 95% percentile bootstrap over 1,000 resamples; comparisons between
models are **paired** bootstraps on shared resamples, so the shared difficulty of the images
cancels out.

| Run | Head | Augmentation | Test QWK | 95% CI | Accuracy | Balanced acc. |
|---|---|---|---:|---|---:|---:|
| `densenet121_ordinal_aug` | ordinal | strong | 0.8910 | [0.869, 0.909] | 0.7564 | 0.6001 |
| **`densenet121_weighted`** | softmax | mild | **0.8896** | [0.862, 0.915] | **0.7964** | **0.6370** |
| `densenet121_ordinal` | ordinal | mild | 0.8896 | [0.868, 0.909] | 0.7527 | 0.5716 |
| `densenet121_aug` | softmax | strong | 0.8824 | [0.856, 0.907] | 0.7655 | 0.6207 |
| `resnet50_weighted` | softmax | mild | 0.8814 | [0.852, 0.909] | 0.8000 | 0.6038 |
| `effnetb3_weighted` | softmax | mild | 0.8583 | [0.824, 0.888] | 0.7800 | 0.6428 |
| `densenet121_long` | softmax | mild | 0.8455 | [0.805, 0.880] | 0.7727 | 0.5927 |

**`densenet121_weighted` is the reported model.** It has the best accuracy and balanced
accuracy, and its QWK is statistically indistinguishable from the top of the table.

Per-class performance of the reported model on the test split:

| Stage | Class | Precision | Recall | F1 | n |
|---:|---|---:|---:|---:|---:|
| 0 | No DR | 0.971 | 0.971 | 0.971 | 271 |
| 1 | Mild | 0.537 | 0.518 | 0.527 | 56 |
| 2 | Moderate | 0.717 | 0.727 | 0.722 | 150 |
| 3 | Severe | 0.297 | 0.379 | 0.333 | 29 |
| 4 | Proliferative DR | 0.722 | 0.591 | 0.650 | 44 |

Exact grade match 79.64%, within one grade 96.18%, mean absolute error 0.251 stages.

**Quadratic Weighted Kappa (QWK) is the primary metric**, not accuracy. Diabetic retinopathy
grading is *ordinal*: predicting stage 4 when the truth is stage 0 is a far worse error than
predicting stage 1. QWK penalises errors by the square of their distance. It is also the
official APTOS 2019 competition metric, making these numbers comparable to published work.

The argument in one sentence: *a degenerate classifier predicting "No DR" for every image
scores roughly 49% accuracy on this dataset and a QWK of 0.*

### The main finding: validation-selected gains did not transfer

The four-arm ablation was designed to test two changes to the baseline — an ordinal
regression head in place of the five-way softmax, and a stronger augmentation regime — with
one variable per arm and a shared schedule.

On **validation**, the ordinal head looked like a clear win:

| Comparison (validation QWK) | Difference | 95% CI | Distinguishable |
|---|---:|---|---|
| ordinal vs `densenet121_weighted` | +0.0202 | [0.0042, 0.0377] | **yes** |
| ordinal vs `resnet50_weighted` | +0.0245 | [0.0055, 0.0449] | **yes** |
| ordinal vs `effnetb3_weighted` | +0.0256 | [0.0061, 0.0468] | **yes** |

On the **held-out test split**, the same comparison is **−0.0001, 95% CI [−0.0236, 0.0240]**.
The entire advantage disappeared.

The cause is measurable rather than speculative. The ordinal head fits four decision
thresholds on the 550-image validation split; the softmax arms fit none. Validation-to-test
drop, per run:

| Run | Val QWK | Test QWK | Drop |
|---|---:|---:|---:|
| `densenet121_weighted` | 0.8992 | 0.8896 | −0.010 |
| `resnet50_weighted` | 0.8949 | 0.8814 | −0.014 |
| `densenet121_ordinal` | 0.9194 | 0.8896 | −0.030 |
| `densenet121_ordinal_aug` | 0.9200 | 0.8910 | −0.029 |
| `densenet121_long` | 0.9135 | 0.8455 | −0.068 |

The arms carrying validation-fitted parameters drop roughly three times as far. And
`densenet121_long`, second-best on validation, finished **last** on test.

The conclusion this project draws is about the *procedure*, not the architecture: with 550
images in each of the validation and test splits, model selection is dominated by sampling
noise. Five-fold cross-validation would be the correct remedy and is the first item under
[Future work](#future-work).

### The one change that did survive

The ordinal head eliminated every catastrophic error:

| Model | 0 grades out | 1 | 2 | **3** | Within two grades |
|---|---:|---:|---:|---:|---:|
| `densenet121_weighted` | 438 | 91 | 16 | **5** | 99.1% |
| `densenet121_ordinal` | 414 | 114 | 22 | **0** | **100%** |

A regression head trained on distance cannot produce a prediction three grades from the
truth. It buys that at the cost of exact-match accuracy (0.796 → 0.753). For a
screening-support tool, "never off by three stages" is arguably the better property, and the
trade is stated here rather than buried.

### On the architecture comparison

All three architectures were trained on identical data (same stratified split, verified by a
shared split fingerprint), identical preprocessing, augmentation, loss and optimisation
schedule. The only variable was the architecture.

The spread across architectures is smaller than the width of a single confidence interval.
Paired comparisons in `reports/m6_test_pairwise.csv` show only `densenet121` vs
`efficientnet_b3` separating from zero; every other pair overlaps.

Worth noting: **DenseNet121 achieved the highest QWK with under a third of ResNet50's
parameters**, which suggests that on a dataset of this size the limiting factor is training
data volume rather than model capacity.

> **Reproducibility note.** An earlier evaluation of the same `densenet121_weighted`
> checkpoint reported QWK 0.8910 against the 0.8896 above. The difference is a handful of
> borderline images flipping under non-deterministic mixed-precision inference. The tables
> here are regenerated from `reports/m6_test_comparison.csv`; no number in this README was
> typed by hand.

---

## What the system does

```
┌──────────────────────── PRESENTATION ────────────────────────┐
│  Streamlit app                                               │
│   New screening · Patient timeline · Records · About         │
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
│  → weighted loss → checkpoint + metrics                      │
└──────────────────────────────────────────────────────────────┘
```

| Capability | Detail |
|---|---|
| Severity grading | 5 classes: No DR, Mild, Moderate, Severe, Proliferative DR |
| Explainability | Grad-CAM on the final convolutional layer, with border-attention diagnostics |
| Visit records | SQLite: patient ID, date, predicted stage, confidence, full probability vector, image and heatmap paths, model version |
| Progression | Trend classification (improving / stable / worsening), least-squares slope in stages per year, step chart |
| Reporting | Downloadable per-patient PDF, disclaimer first |

The interface is written for a reader with no medical background. Every result leads with a
sentence in ordinary English — what was found and what a person would usually do about it —
with the probabilities, heatmap diagnostics and checkpoint provenance directly below rather
than hidden. Severity colour never carries meaning alone: the stage number and its name
travel with it everywhere, and text contrast against every background was computed rather
than eyeballed.

The app handles both head types. Selecting an ordinal checkpoint produces a single severity
score, which the interface reports as such — the five bars shown are derived from that one
number, and the page says so instead of presenting them as a learned distribution.

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

Two regimes are compared. Geometry is generous, colour is deliberately conservative in both.

| | Mild (baseline) | Strong (`*_aug` arms) |
|---|---|---|
| Horizontal / vertical flip | p = 0.5 each | p = 0.5 each |
| Rotation | ±15° | **±180°** |
| Scale | 0.90–1.00 | **0.70–1.00** |
| Random erasing | — | **p = 0.25** |
| Brightness / contrast / saturation | 0.15 / 0.15 / 0.10 | 0.15 / 0.15 / 0.10 |
| **Hue** | **0.0 — unchanged** | **0.0 — unchanged** |

A fundus photograph has no canonical orientation (cameras rotate; left and right eyes are
mirror images), so flips and rotation are free label-preserving variety — which is why the
strong regime uses the full 360°. Colour is different: haemorrhages are dark red and exudates
pale yellow, so lesion identity is partly *in* the colour. Shifting hue would corrupt the
label while leaving the image looking plausible.

The strong regime was motivated by measurement, not taste. Every baseline run overfits: the
ratio of validation loss to training loss over the final five epochs was 5.16× for
`densenet121_long`. Strong augmentation brought that to **3.36×** — it did what it was
designed to do — but the improvement did not reach the aggregate metrics.

### Split

Stratified 70 / 15 / 15 (2,562 / 550 / 550), seed 42, written to `data/splits/` and
committed. Every run reads those CSVs and records a **split fingerprint**, so two runs can be
proven to have trained on identical data before their scores are compared.

### Two prediction heads

| | `classification` | `ordinal` |
|---|---|---|
| Output | 5 logits → softmax | 1 regression score on the 0–4 scale |
| Loss | Weighted cross-entropy | Weighted Smooth L1 |
| Decision | `argmax` | four thresholds, fitted on **validation** by coordinate ascent |
| Stored in checkpoint | class names | class names **and the thresholds** |

The ordinal head exists because the error analysis demanded it: the baseline is within one
grade 96% of the time but exact only 80% of the time, and a five-way softmax has no notion
that stage 2 lies between 1 and 3. Smooth L1 rather than plain MSE, because clinician-graded
labels are noisy and squared error lets one mislabelled image dominate a batch.

Thresholds are part of the model, not a post-hoc convenience: they are fitted on validation,
written into the checkpoint, and applied unchanged at test time. `src/models/thresholds.py`
documents why fitting them for QWK, accuracy or balanced accuracy produces genuinely
different cut-points.

### Training

| | |
|---|---|
| Transfer learning | ImageNet-pretrained backbones via `timm` |
| Loss | Weighted, from inverse class frequency **on the train split only** |
| Optimiser | AdamW, cosine schedule with 1 epoch linear warmup |
| Epochs | 45 |
| Early stopping | Patience 12, on a **3-epoch moving average** of validation QWK |
| Checkpoint selection | Best single epoch by raw validation QWK |
| Precision | Mixed (AMP) |
| Effective batch | 16–18 (gradient accumulation) |
| Hardware | NVIDIA RTX 3050 6 GB Laptop GPU, CUDA 12.1 |

Class weights are computed from the training split alone; using whole-dataset counts would
leak test-set label statistics into training.

Early stopping is judged on a smoothed curve because validation QWK over 550 images carries
roughly ±0.03 of sampling noise. The original settings (20 epochs, patience 5 on raw scores)
stopped the EfficientNet-B3 run at epoch 8 having peaked at epoch 3 — before the cosine
schedule had annealed at all.

---

## Beyond the baseline

Three additions that need **no GPU and no retraining**, because Milestone 6 saves every run's
probability output to `experiments/<run>/{split}_predictions.npz`.

### Ensembling

Averaging the three architectures' probabilities. The members disagree on **23.3%** of test
images, and on 89% of those contested images at least one member holds the correct grade —
that is the headroom an ensemble could recover. Geometric-mean ensembling reached test QWK
0.8935 against 0.8910 for the best single model; the paired bootstrap difference is −0.0025,
95% CI [−0.025, 0.021]. Reported as *no worse*, not as better.

### Calibration

Every model is overconfident — the Grad-CAM table contains predictions at 0.9999921
confidence from a model that is right 80% of the time. Temperature scaling, fitted on
validation only, gives temperatures of 1.26–1.43 and cannot change any prediction, since
dividing logits by a positive constant preserves their order.

ECE improves for DenseNet (0.063 → 0.033) and ResNet (0.072 → 0.047), but **every ECE change
has a bootstrap interval containing zero** — with 15 bins over 550 images, ECE cannot resolve
a change that size. NLL, which needs no binning, improves for all four models. The app uses
the calibrated wording rather than the raw percentage.

### Ordinal decoding without retraining

A softmax over an ordinal scale already contains a continuous estimate: its expected value,
`Σ k·p_k`. Cutting that with validation-fitted thresholds turns any existing classifier into
a grader. `reports/m6c_objective_tradeoff.csv` measures the objective choice, and it is a
real trade — on the ensemble:

| Thresholds fitted for | QWK | Accuracy | Balanced acc. | Severe recall |
|---|---:|---:|---:|---:|
| *(none — argmax)* | 0.8935 | 0.8036 | 0.6383 | 0.414 |
| accuracy | 0.9011 | **0.8164** | 0.5931 | 0.241 |
| QWK | **0.9058** | 0.7800 | 0.6253 | 0.586 |
| balanced accuracy | 0.8994 | 0.7691 | **0.6710** | **0.690** |

Optimising exact accuracy nearly halves Severe recall; optimising balanced accuracy nearly
doubles it. For a screening tool the third row is the defensible choice, and the cost in
exact accuracy should be stated rather than hidden.

### Test-time augmentation

Averaging over the four flip views. Measured on validation across all seven runs, the effect
ranged from −0.006 to +0.017 QWK and was **negative for the selected model**. Not adopted —
reported as a measured negative result. See `reports/m6_tta_effect.csv`.

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
python -m scripts.m2_cache_images --workers 4   # cache 300x300 images
python -m scripts.m3_split                      # stratified split (run once)
```

Architecture comparison:

```powershell
python -m scripts.m4_train --experiment resnet50
python -m scripts.m4_train --experiment efficientnet_b3
python -m scripts.m4_train --experiment densenet121
```

The four-arm ablation — one variable each, shared schedule and control:

```powershell
python -m scripts.m4_train --experiment densenet121_long          # control
python -m scripts.m4_train --experiment densenet121_aug           # + strong augmentation
python -m scripts.m4_train --experiment densenet121_ordinal       # + ordinal head
python -m scripts.m4_train --experiment densenet121_ordinal_aug   # both
```

Roughly 20–35 minutes per run on an RTX 3050. Smoke-test first — two epochs on 64 images,
saves nothing, and exercises every code path:

```powershell
python -m scripts.m4_train --experiment densenet121_ordinal --smoke
```

Selection, then evaluation:

```powershell
python -m scripts.m6_evaluate --split val       # choose here
python -m scripts.m6_evaluate --split val --tta # decide on TTA here
python -m scripts.m6_evaluate                   # test: once, after choosing

python -m scripts.m6b_refine                              # ensemble + calibration (no GPU)
python -m scripts.m6c_ordinal_decode --metric balanced_accuracy   # thresholds (no GPU)

python -m scripts.m7_gradcam                    # explainability figures
python -m scripts.m8_init_db --demo             # database + simulated histories
python -m scripts.m9_progression                # progression charts
streamlit run app/app.py                        # the application
```

**The test split is evaluated once**, after every choice — architecture, arm, TTA, threshold
objective — has been made on validation. The [main finding](#the-main-finding-validation-selected-gains-did-not-transfer)
is what happens when that discipline is followed and the validation-selected winner still
fails to transfer.

### 4. Tests

```powershell
pytest -q
```

43 tests covering dataset integrity checks, split correctness and leakage detection, the
database layer, and progression logic.

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
├── .streamlit/config.toml      app theme (light, warm) — not CSS overrides
├── configs/
│   ├── config.yaml             all paths and hyper-parameters
│   └── experiments/            one override file per run
├── data/splits/                the exact train/val/test partition (committed)
├── docs/                       architecture notes, quickstart, references
├── experiments/<run>/          per run: metrics, history, config snapshot, predictions, logs
├── notebooks/                  four milestone notebooks
├── reports/
│   ├── figures/                every figure, all generated
│   └── *.csv, *.json           every results table
├── scripts/                    one entry point per milestone step
├── src/
│   ├── data/                   inspection, preprocessing, transforms, dataset, split
│   ├── models/                 factory, training, metrics, evaluation, bootstrap, figures,
│   │                           ensemble, calibrate, thresholds
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
| `*.pt` model checkpoints | ~28 MB each, 267 MB in total; bloats every clone | Attached to the GitHub **Releases** page, or retrain with the commands above |
| `app/assets/*.sqlite3` | Runtime state, regenerable | `python -m scripts.m8_init_db --demo` |
| `kaggle.json` | Credential | Yours, from your Kaggle account |

Everything needed to reproduce the results — code, configs, the exact data split, every
run's saved predictions, and all generated metrics — **is** committed. The `.npz` prediction
files are what let `m6b_refine` and `m6c_ordinal_decode` be re-run on any laptop with no GPU.

---

## Limitations

Stated plainly, because a screening tool that overstates itself is worse than no tool.

- **Model selection is dominated by noise at this sample size.** With 550 images in each of
  the validation and test splits, a validation advantage whose confidence interval excluded
  zero vanished entirely on test, and the second-best validation arm finished last. This is
  the project's central methodological finding and it limits every comparison in it.
- **Single dataset, no external validation.** Trained and evaluated entirely on APTOS 2019.
  Performance on images from different cameras, populations, or capture protocols is unknown.
- **Single seed.** Each configuration was trained once. Run-to-run variance is not measured,
  and is plausibly of the same size as the differences between arms.
- **Small rare classes.** Severe DR is 5.3% of the data — 29 images in the test split. Its
  per-class metrics carry wide uncertainty, and it is the stage the model gets wrong most.
- **No clinician review.** No ophthalmologist validated any prediction or heatmap.
- **Simulated longitudinal data.** APTOS is cross-sectional: one image per patient, no
  follow-up. The progression module is demonstrated on **synthetic** visit histories. Every
  simulated record is flagged `is_simulated=1` in the database, marked in the UI and on every
  chart, and labelled in the PDF report. The progression *logic* is real and would run
  unchanged on real longitudinal data.
- **Grad-CAM localises regions, not lesions.** The heatmap is roughly 10×10 upsampled to
  300×300. A plausible-looking heatmap is not proof the model reasoned correctly — and the
  heatmap statistics record cases where a large share of attention sits on the image border
  rather than the retina.
- **Ordinal probabilities are derived, not learned.** The five-way bars shown for an ordinal
  checkpoint are a re-expression of one score. Their AUC is indicative and is not comparable
  to a softmax model's.
- **Suggested review intervals are illustrative only** and are not clinical guidance.
- **Not deployed publicly, deliberately.** A publicly reachable tool that returns a severity
  grade for an uploaded retinal photograph carries a real risk of misuse regardless of any
  disclaimer.

---

## Future work

Several items from the original list were attempted during this project; their outcomes are
recorded above rather than left as aspirations.

| Idea | Status |
|---|---|
| Ordinal regression loss matching the metric | **Tried.** Won on validation, did not transfer to test. Eliminated all 3-grade errors. |
| Test-time augmentation | **Tried.** −0.006 to +0.017 QWK on validation; negative for the selected model. Not adopted. |
| Ensembling | **Tried.** No worse than the best single model; the difference is within noise. |
| Stronger augmentation | **Tried.** Cut the overfitting gap from 5.16× to 3.36× without moving the aggregate metrics. |
| Confidence calibration | **Tried.** Temperatures 1.26–1.43; NLL improves, ECE cannot resolve the change at this sample size. |

What the results actually point at next:

- **Five-fold cross-validation.** The single highest-value change. It would use all 3,662
  images for training across folds, give five models to ensemble properly, and — most
  importantly — replace single-split selection with an estimate that has a variance.
- **Multiple seeds per configuration**, so run-to-run variance can be separated from the
  effect of an intervention.
- **External validation** on IDRiD or Messidor-2 — inference only, no retraining — to turn
  "no external validation" from a limitation into a measurement.
- **Pretraining on EyePACS** (Kaggle DR 2015, ~35,000 images on the same scale), then
  fine-tuning on APTOS.
- **Higher input resolution.** Microaneurysms are a few pixels across at 300×300; 448 or 512
  fits on 6 GB at batch 4 with gradient accumulation.
- **Ben Graham preprocessing** (local-average subtraction) and a tighter circular retina
  mask, which would also address the border-attention finding.
- Lesion-level annotation to validate Grad-CAM against ground-truth pathology locations.

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
10. C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, "On calibration of modern neural networks,"
    in *Proc. ICML*, 2017, pp. 1321–1330.
11. R. Wightman, "PyTorch Image Models (timm)," GitHub, 2019.

---

## Licence and use

Academic coursework. The APTOS 2019 dataset remains subject to Kaggle's competition rules.
This software is not licensed, certified, or approved for any clinical purpose.
