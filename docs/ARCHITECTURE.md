# Architecture and Data Flow

## 1. System overview

The system has three layers that are deliberately kept separate, so each can be
tested, demonstrated and written up on its own.

```
┌──────────────────────────── PRESENTATION LAYER ────────────────────────────┐
│  Streamlit app (app/)                                                      │
│   • Upload page    • Prediction + confidence   • Grad-CAM viewer           │
│   • Patient history page   • Progression chart   • PDF report download     │
└───────────────▲────────────────────────────────────────────▲───────────────┘
                │                                            │
        inference request                             history queries
                │                                            │
┌───────────────┴──────────── APPLICATION LAYER ─────────────┴───────────────┐
│  Inference service (src/models/predict.py)   Progression analysis          │
│   • load checkpoint (once, cached)            (src/analysis/progression.py)│
│   • preprocess → forward pass → softmax       • trend classification        │
│   • Grad-CAM (src/explain/gradcam.py)         • chart builders             │
│                                                                            │
│  Data-access layer (src/db/dao.py)  ── SQLite ──►  app/assets/*.sqlite3    │
└───────────────▲────────────────────────────────────────────────────────────┘
                │
┌───────────────┴──────────────── MODEL LAYER ───────────────────────────────┐
│  Training pipeline (offline, run once per experiment)                      │
│   APTOS 2019 → preprocess → stratified split → DataLoader → EfficientNet-B3│
│   → weighted CE loss → checkpoint (experiments/<run>/best.pt) + metrics.json│
└────────────────────────────────────────────────────────────────────────────┘
```

## 2. Offline training data flow (Milestones 1–7)

```
data/raw/aptos2019/train.csv + train_images/*.png
        │
        │  M1  integrity check, class distribution, image statistics
        ▼
   reports/figures/*.png, reports/m1_*.csv
        │
        │  M2  crop black borders → resize → [optional CLAHE] → normalise (ImageNet)
        ▼                 augment (train only): flip, rotate, brightness/contrast, shift-scale
   data/interim/ (cached 300×300 images, optional but ~5× faster epochs)
        │
        │  M3  stratified split, seed=42
        ▼
   data/splits/{train,val,test}.csv     70 / 15 / 15, class ratios preserved
        │
        │  M4/M5  DataLoader → EfficientNet-B3 (timm, ImageNet weights)
        │         head replaced with Linear(in_features → 5)
        │         weighted cross-entropy + optional WeightedRandomSampler
        │         AdamW + cosine schedule + AMP + early stopping on val QWK
        ▼
   experiments/<run_name>/{best.pt, config_snapshot.yaml, history.csv, metrics.json}
        │
        │  M6  test-set evaluation
        ▼
   accuracy, per-class precision/recall/F1, confusion matrix,
   Quadratic Weighted Kappa, one-vs-rest ROC-AUC
        │
        │  M7  Grad-CAM on the final conv block
        ▼
   heatmap overlays proving the model attends to lesions, not artefacts
```

## 3. Online inference data flow (Milestones 8–10)

```
User uploads fundus image + patient ID + visit date
        │
        ▼
 preprocess (EXACTLY the validation transform — no augmentation)
        │
        ▼
 EfficientNet-B3 forward pass  ──►  logits ──► softmax ──► (stage, confidence)
        │
        ├──► Grad-CAM overlay saved to app/assets/gradcam/<uuid>.png
        ├──► source image saved to app/assets/uploads/<uuid>.png
        │
        ▼
 INSERT INTO visits (patient_id, visit_date, predicted_stage, confidence,
                     image_path, gradcam_path, model_version, created_at)
        │
        ▼
 Patient history page: SELECT … WHERE patient_id = ? ORDER BY visit_date
        │
        ▼
 Progression analysis: stage over time → trend (improving / stable / worsening)
        │
        ▼
 Plotly step chart + downloadable PDF report (with disclaimer banner)
```

## 4. Why these design choices

| Decision | Reason |
|---|---|
| **EfficientNet-B3 at 300×300** | Best accuracy-per-FLOP in its class; 300×300 is its native training resolution. Fine lesions (microaneurysms) are only a few pixels wide, so resolution matters more than depth here. ResNet50 at 224×224 is kept as a comparison baseline for the report. |
| **Transfer learning, not from scratch** | 3.6k images is far too few to learn low-level vascular texture filters from random init. ImageNet features transfer well to fundus imagery. |
| **Crop black borders before resize** | Raw fundus photos are a circle on a black rectangle. Resizing without cropping wastes 30–50 % of the input on black pixels and changes effective magnification per image. |
| **Weighted CE + oversampling as *separate* experiments** | So the report can show an ablation table rather than one unexplained number. |
| **Quadratic Weighted Kappa as the headline metric** | DR grading is *ordinal*: predicting 4 when the truth is 0 is much worse than predicting 1. Plain accuracy ignores this; QWK is also the official APTOS competition metric, so your results are comparable to published work. |
| **Grad-CAM** | A 5-class softmax is not trustworthy on its own. Heatmaps let a human check the model looked at haemorrhages/exudates rather than lens flare or the image border — this is the "explainable" in the project title. |
| **SQLite** | Zero-configuration, single file, ships with Python. Adequate for a single-user prototype and trivially demonstrable to an examiner. |
| **Streamlit** | Pure Python; no HTML/JS needed; renders images and Plotly charts natively — the fastest route to a demo-able UI for an ML project. |
| **Simulated visit histories** | APTOS has no longitudinal patient data. Multi-visit progression is therefore demonstrated with clearly-labelled simulated histories; every simulated record is flagged in the database and the UI. |

## 5. Database schema (implemented in Milestone 8)

```sql
patients(patient_id TEXT PK, display_name TEXT, notes TEXT, created_at TEXT)

visits(visit_id INTEGER PK AUTOINCREMENT,
       patient_id TEXT NOT NULL REFERENCES patients(patient_id),
       visit_date TEXT NOT NULL,           -- ISO-8601 'YYYY-MM-DD'
       predicted_stage INTEGER NOT NULL CHECK(predicted_stage BETWEEN 0 AND 4),
       confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
       probabilities TEXT,                 -- JSON array of 5 floats
       image_path TEXT NOT NULL,
       gradcam_path TEXT,
       model_version TEXT,
       is_simulated INTEGER NOT NULL DEFAULT 0,
       created_at TEXT NOT NULL)
```

## 6. Module responsibilities

| Path | Responsibility |
|---|---|
| `src/utils/config.py` | Load `configs/config.yaml`, resolve absolute paths |
| `src/utils/seed.py` | Seed Python/NumPy/PyTorch; optional full determinism |
| `src/data/inspect.py` | Integrity checks, class distribution, image statistics |
| `src/data/preprocess.py` | Border cropping, CLAHE, resize, normalisation *(M2)* |
| `src/data/transforms.py` | Albumentations train/val pipelines *(M2)* |
| `src/data/dataset.py` | `torch.utils.data.Dataset` for APTOS *(M4)* |
| `src/data/split.py` | Seeded stratified split *(M3)* |
| `src/models/factory.py` | Backbone construction + head replacement *(M4)* |
| `src/models/train.py` | Training loop, AMP, scheduler, checkpointing *(M4/M5)* |
| `src/models/evaluate.py` | Metrics + confusion matrix + ROC *(M6)* |
| `src/models/predict.py` | Single-image inference service for the app *(M10)* |
| `src/explain/gradcam.py` | Grad-CAM hooks and overlay rendering *(M7)* |
| `src/db/schema.py`, `src/db/dao.py` | SQLite schema and CRUD *(M8)* |
| `src/analysis/progression.py` | Trend logic and chart builders *(M9)* |
| `app/app.py`, `app/pages/*` | Streamlit UI *(M10)* |
