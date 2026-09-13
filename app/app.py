"""
Streamlit web application.
File location: <project_root>/app/app.py

Run:
    streamlit run app/app.py

Four pages:
    Screen           upload a fundus image -> prediction + confidence + Grad-CAM -> save visit
    Patient history  visit table, progression chart, trend summary, PDF report
    Database         statistics, recent visits, CSV export
    About            model provenance, metrics, limitations

Design note
-----------
Inference uses `build_transforms(cfg, train=False)` and `preprocess_from_config`
- the exact same functions training used. That is deliberate: an app with its
own copy of the preprocessing drifts from the trained model and predicts
differently from what your evaluation measured, which is a miserable bug to
find the night before a demo.
"""

from __future__ import annotations

import io
import json
import sys
from datetime import date, datetime
from pathlib import Path

# --- make `src` importable ---------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.analysis.progression import (
    analyse_progression,
    next_review_suggestion,
    progression_figure,
    progression_summary_text,
    simulate_visit_history,
)
from src.data.preprocess import bgr_to_rgb, preprocess_from_config
from src.db import dao
from src.explain.gradcam import cam_for_image
from src.models.evaluate import find_runs, load_checkpoint
from src.utils.config import class_names, get_path, load_config

# ---------------------------------------------------------------------------
st.set_page_config(page_title="DR Screening Support Prototype",
                   page_icon="👁", layout="wide")

STAGE_COLOURS = {0: "#2E7D32", 1: "#9E9D24", 2: "#EF6C00", 3: "#D84315", 4: "#B71C1C"}


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_config():
    cfg = load_config()
    cfg["training"]["num_workers"] = 0     # the app never uses DataLoader workers
    return cfg


@st.cache_resource
def get_connection(db_path: str):
    conn = dao.connect(db_path)
    dao.init_db(conn)
    return conn


@st.cache_resource
def get_model(run_name: str, _cfg):
    """Load a checkpoint once per session. The leading underscore on _cfg tells
    Streamlit not to try hashing the config dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = get_path(_cfg, "experiments_dir") / run_name
    model, ckpt = load_checkpoint(run_dir / "best.pt", device,
                                  num_classes=_cfg["dataset"]["num_classes"])
    return model, ckpt, device


def available_runs(cfg):
    return [d.name for d in find_runs(get_path(cfg, "experiments_dir"))]


def best_run_name(cfg) -> str | None:
    """Pick the run with the highest test QWK, falling back to validation."""
    scored = []
    for run_dir in find_runs(get_path(cfg, "experiments_dir")):
        for fname, key in (("metrics_test.json", "qwk"), ("metrics_val.json", None)):
            path = run_dir / fname
            if not path.is_file():
                continue
            data = json.loads(path.read_text())
            score = data.get("qwk") if key else data.get("val_metrics", {}).get("qwk")
            if score is not None:
                scored.append((score, run_dir.name))
                break
    if not scored:
        runs = available_runs(cfg)
        return runs[0] if runs else None
    return max(scored)[1]


# ---------------------------------------------------------------------------
# Shared UI
# ---------------------------------------------------------------------------
def disclaimer_banner(cfg):
    st.error(
        "**ACADEMIC PROTOTYPE - NOT A MEDICAL DEVICE.** "
        + " ".join(cfg["app"]["disclaimer"].split())
    )


def stage_badge(stage: int, names, confidence: float | None = None):
    colour = STAGE_COLOURS.get(int(stage), "#555")
    extra = f" &nbsp;|&nbsp; confidence {confidence:.1%}" if confidence is not None else ""
    st.markdown(
        f"<div style='background:{colour};color:white;padding:14px 18px;"
        f"border-radius:8px;font-size:20px;font-weight:600;'>"
        f"Stage {stage} — {names[int(stage)]}{extra}</div>",
        unsafe_allow_html=True,
    )


def probability_table(probabilities, names) -> pd.DataFrame:
    return pd.DataFrame({
        "stage": range(len(names)),
        "class": names,
        "probability": [round(float(p), 4) for p in probabilities],
    })


# ---------------------------------------------------------------------------
# Page: Screen
# ---------------------------------------------------------------------------
def page_screen(cfg, conn, names):
    st.header("Screen a fundus image")

    runs = available_runs(cfg)
    if not runs:
        st.warning("No trained model found in `experiments/`. Train a model first "
                   "(Milestones 4-5).")
        return

    col_a, col_b = st.columns([2, 1])
    with col_b:
        default = best_run_name(cfg)
        run_name = st.selectbox("Model", runs,
                                index=runs.index(default) if default in runs else 0)
        patient_id = st.text_input("Patient ID", value="P001",
                                   help="Any identifier. Do not enter real patient data.")
        visit_date = st.date_input("Visit date", value=date.today())
        save_record = st.checkbox("Save this prediction to the database", value=True)

    with col_a:
        uploaded = st.file_uploader("Fundus image", type=["png", "jpg", "jpeg", "tif", "tiff"])

    if uploaded is None:
        st.info("Upload a retinal fundus photograph to begin. "
                "Images from `data/raw/train_images/` work for a demonstration.")
        return

    try:
        buffer = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
        image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image_bgr is None:
            st.error("Could not decode that file as an image.")
            return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read the upload: {exc}")
        return

    with st.spinner("Loading model..."):
        try:
            model, ckpt, device = get_model(run_name, cfg)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load the model: {exc}")
            return

    # Exactly the preprocessing used in training - see the module docstring.
    processed_rgb = bgr_to_rgb(preprocess_from_config(image_bgr, cfg))

    with st.spinner("Predicting and generating Grad-CAM..."):
        try:
            result = cam_for_image(model, processed_rgb, cfg, device)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Inference failed: {exc}")
            return

    st.divider()
    stage_badge(result["predicted"], names, result["confidence"])
    st.caption(f"Model: `{ckpt.get('backbone')}` from run `{run_name}` "
               f"(checkpoint epoch {ckpt.get('epoch')})")

    c1, c2, c3 = st.columns(3)
    c1.image(bgr_to_rgb(image_bgr), caption="Uploaded image", use_container_width=True)
    c2.image(processed_rgb, caption=f"Preprocessed {processed_rgb.shape[1]}x{processed_rgb.shape[0]}",
             use_container_width=True)
    c3.image(result["overlay"], caption="Grad-CAM (red = most influential)",
             use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Class probabilities")
        table = probability_table(result["probabilities"], names)
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.bar_chart(table.set_index("class")["probability"])

    with right:
        st.subheader("Heatmap diagnostics")
        stats = result["stats"]
        st.metric("Attention in image border", f"{stats['border_fraction']:.1%}")
        if stats["border_fraction"] > 0.35:
            st.warning("A large share of attention is on the image border rather than "
                       "the retina. Treat this prediction with extra caution.")
        st.json(stats)

    # -- save ---------------------------------------------------------------
    if save_record:
        uploads_dir = get_path(cfg, "uploads_dir")
        gradcam_dir = get_path(cfg, "gradcam_dir")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        gradcam_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{patient_id}_{stamp}"
        image_path = uploads_dir / f"{stem}.png"
        cam_path = gradcam_dir / f"{stem}_cam.png"

        cv2.imwrite(str(image_path), cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(cam_path), cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR))

        if st.button("Save visit record", type="primary"):
            try:
                visit_id = dao.add_visit(
                    conn, patient_id=patient_id, visit_date=visit_date,
                    predicted_stage=result["predicted"], confidence=result["confidence"],
                    probabilities=result["probabilities"],
                    image_path=str(image_path), gradcam_path=str(cam_path),
                    model_version=run_name, is_simulated=False,
                )
                st.success(f"Saved as visit #{visit_id} for patient {patient_id}.")
            except ValueError as exc:
                st.error(f"Could not save: {exc}")


# ---------------------------------------------------------------------------
# Page: Patient history
# ---------------------------------------------------------------------------
def page_history(cfg, conn, names):
    st.header("Patient history and progression")

    patients = dao.list_patients(conn)
    if patients.empty:
        st.info("No patients yet. Screen an image, or generate a simulated history below.")
        _simulation_controls(conn)
        return

    ids = patients["patient_id"].tolist()
    patient_id = st.selectbox("Patient", ids)

    visits = dao.get_visits(conn, patient_id)
    if visits.empty:
        st.warning("This patient has no visits.")
        _simulation_controls(conn)
        return

    if visits["is_simulated"].any():
        st.warning(
            f"**{int(visits['is_simulated'].sum())} of {len(visits)} visits are SIMULATED.** "
            "APTOS 2019 contains no longitudinal data, so multi-visit histories are "
            "synthetic and exist only to demonstrate the progression module."
        )

    analysis = analyse_progression(visits)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Visits", analysis["n_visits"])
    m2.metric("Latest stage", f"{analysis['latest_stage']} — {names[analysis['latest_stage']]}"
              if analysis["latest_stage"] is not None else "-")
    m3.metric("Trend", analysis["trend"].title())
    m4.metric("Change", f"{analysis['stage_change']:+d}"
              if analysis["stage_change"] is not None else "-")

    st.info(progression_summary_text(analysis, names))

    if analysis["latest_stage"] is not None:
        suggestion = next_review_suggestion(analysis["latest_stage"], analysis["trend"])
        st.caption(f"Illustrative review interval: {suggestion['months']} months "
                   f"(~{suggestion['approx_date']}). {suggestion['disclaimer']}")

    if len(visits) >= 1:
        st.pyplot(progression_figure(visits, names, patient_id))

    st.subheader("Visit records")
    display = visits[["visit_id", "visit_date", "predicted_stage", "confidence",
                      "model_version", "is_simulated"]].copy()
    display["stage"] = display["predicted_stage"].map(lambda s: f"{s} — {names[int(s)]}")
    st.dataframe(display.drop(columns=["predicted_stage"]),
                 hide_index=True, use_container_width=True)

    with st.expander("Stored images and Grad-CAM overlays"):
        for _, row in visits.iterrows():
            cols = st.columns([1, 1, 2])
            for col, key, label in ((cols[0], "image_path", "image"),
                                    (cols[1], "gradcam_path", "Grad-CAM")):
                path = row.get(key)
                if path and Path(path).is_file():
                    col.image(str(path), caption=f"{row['visit_date']} {label}",
                              use_container_width=True)
                else:
                    col.caption(f"no {label} stored")
            cols[2].write(f"**Visit {row['visit_id']}** — {row['visit_date']} — "
                          f"stage {row['predicted_stage']} ({row['confidence']:.1%})")

    pdf = build_pdf_report(cfg, patient_id, visits, analysis, names)
    if pdf is not None:
        st.download_button("Download PDF report", data=pdf,
                           file_name=f"DR_report_{patient_id}.pdf",
                           mime="application/pdf", type="primary")

    st.divider()
    _simulation_controls(conn)


def _simulation_controls(conn):
    with st.expander("Generate a SIMULATED visit history (for demonstration)"):
        st.caption("APTOS has no longitudinal data. These records are synthetic, are "
                   "flagged in the database, and must be described as simulated in any "
                   "report or demonstration.")
        c1, c2, c3, c4 = st.columns(4)
        sim_id = c1.text_input("Patient ID", value="SIM001")
        n_visits = c2.slider("Visits", 2, 8, 5)
        pattern = c3.selectbox("Pattern", ["worsening", "improving", "stable", "fluctuating"])
        start_stage = c4.slider("Starting stage", 0, 4, 1)

        if st.button("Generate simulated history"):
            records = simulate_visit_history(sim_id, n_visits=n_visits,
                                             start_stage=start_stage, pattern=pattern)
            for record in records:
                dao.add_visit(conn, model_version="SIMULATED", **record)
            st.success(f"Created {len(records)} simulated visits for {sim_id}.")
            st.rerun()


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------
def build_pdf_report(cfg, patient_id, visits, analysis, names):
    """One-page PDF. The disclaimer is the first thing on it, by design."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas as pdf_canvas
    except ImportError:
        st.caption("Install reportlab to enable PDF export:  pip install reportlab")
        return None

    buffer = io.BytesIO()
    pdf = pdf_canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 22 * mm

    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(20 * mm, y, "Diabetic Retinopathy Screening Report")
    y -= 7 * mm

    pdf.setFillColorRGB(0.75, 0.1, 0.1)
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(20 * mm, y, "ACADEMIC PROTOTYPE - NOT A MEDICAL DEVICE")
    y -= 5 * mm
    pdf.setFont("Helvetica", 7.5)
    for line in _wrap(" ".join(cfg["app"]["disclaimer"].split()), 118):
        pdf.drawString(20 * mm, y, line)
        y -= 3.6 * mm
    pdf.setFillColorRGB(0, 0, 0)
    y -= 4 * mm

    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(20 * mm, y, f"Patient: {patient_id}")
    y -= 6 * mm

    pdf.setFont("Helvetica", 9)
    for label, value in (
        ("Report generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Visits recorded", str(analysis["n_visits"])),
        ("Latest stage", f"{analysis['latest_stage']} - {names[analysis['latest_stage']]}"
         if analysis["latest_stage"] is not None else "-"),
        ("Trend", analysis["trend"]),
        ("Observation period", f"{analysis['days_observed']} days"
         if analysis["days_observed"] else "-"),
        ("Simulated data included", "YES" if analysis["any_simulated"] else "No"),
    ):
        pdf.drawString(20 * mm, y, f"{label}: {value}")
        y -= 5 * mm

    y -= 3 * mm
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(20 * mm, y, "Summary")
    y -= 5 * mm
    pdf.setFont("Helvetica", 8.5)
    for line in _wrap(progression_summary_text(analysis, names), 105):
        pdf.drawString(20 * mm, y, line)
        y -= 4.2 * mm

    y -= 4 * mm
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(20 * mm, y, "Visit history")
    y -= 6 * mm
    pdf.setFont("Helvetica-Bold", 8)
    for x, head in ((20, "Date"), (55, "Stage"), (110, "Confidence"), (145, "Source")):
        pdf.drawString(x * mm, y, head)
    y -= 4 * mm
    pdf.setFont("Helvetica", 8)
    for _, row in visits.iterrows():
        if y < 25 * mm:
            pdf.showPage(); y = height - 20 * mm; pdf.setFont("Helvetica", 8)
        pdf.drawString(20 * mm, y, str(row["visit_date"]))
        pdf.drawString(55 * mm, y, f"{row['predicted_stage']} - {names[int(row['predicted_stage'])]}")
        pdf.drawString(110 * mm, y, f"{row['confidence']:.1%}")
        pdf.drawString(145 * mm, y, "SIMULATED" if row["is_simulated"] else "recorded")
        y -= 4.5 * mm

    pdf.setFont("Helvetica-Oblique", 7)
    pdf.drawString(20 * mm, 12 * mm,
                   "Generated by an academic screening-support prototype. "
                   "Not validated for clinical use. Not a diagnosis.")
    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


def _wrap(text: str, width: int):
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 <= width:
            current = f"{current} {word}".strip()
        else:
            lines.append(current); current = word
    if current:
        lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# Page: Database
# ---------------------------------------------------------------------------
def page_database(cfg, conn, names):
    st.header("Database")
    stats = dao.database_stats(conn)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", stats["n_patients"])
    c2.metric("Visits", stats["n_visits"])
    c3.metric("Recorded", stats["n_real_visits"])
    c4.metric("Simulated", stats["n_simulated_visits"])

    if stats["n_simulated_visits"]:
        st.warning(f"{stats['n_simulated_visits']} of {stats['n_visits']} visits are "
                   "SIMULATED demonstration data.")

    st.caption(f"File: `{get_path(cfg, 'db_path')}` | schema v{stats['schema_version']}")

    if stats["visits_by_stage"]:
        st.subheader("Visits by predicted stage")
        st.bar_chart(pd.DataFrame({
            "stage": [f"{k} {names[int(k)]}" for k in stats["visits_by_stage"]],
            "count": list(stats["visits_by_stage"].values()),
        }).set_index("stage"))

    st.subheader("Recent visits")
    st.dataframe(dao.recent_visits(conn, 25), hide_index=True, use_container_width=True)

    export = dao.export_all(conn)
    if not export.empty:
        st.download_button("Export all visits as CSV",
                           data=export.to_csv(index=False).encode("utf-8"),
                           file_name="dr_visits_export.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
def page_about(cfg, conn, names):
    st.header("About this prototype")
    st.markdown(f"""
**{cfg['project']['name']}**

Classifies retinal fundus photographs into five diabetic retinopathy severity
stages, explains each prediction with Grad-CAM, and tracks predicted severity
across visits.

| | |
|---|---|
| Dataset | APTOS 2019 Blindness Detection (3,662 labelled images) |
| Input | {cfg['preprocessing']['image_size']}x{cfg['preprocessing']['image_size']} RGB, black borders cropped, ImageNet-normalised |
| Classes | {', '.join(f'{i}: {n}' for i, n in enumerate(names))} |
| Primary metric | Quadratic Weighted Kappa (grading is ordinal) |
| Explainability | Grad-CAM on the final convolutional layer |
| Storage | SQLite ({get_path(cfg, 'db_path').name}) |
| Random seed | {cfg['project']['seed']} |
""")

    st.subheader("Trained models")
    rows = []
    for run_dir in find_runs(get_path(cfg, "experiments_dir")):
        entry = {"run": run_dir.name}
        test_path = run_dir / "metrics_test.json"
        if test_path.is_file():
            data = json.loads(test_path.read_text())
            entry.update(backbone=data.get("backbone"), test_QWK=data.get("qwk"),
                         test_accuracy=data.get("accuracy"),
                         balanced_accuracy=data.get("balanced_accuracy"))
        rows.append(entry)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.caption("No evaluated models yet - run Milestone 6.")

    st.subheader("Limitations")
    st.markdown("""
- Trained and evaluated on a **single dataset** with no external validation.
- The rarest class (Severe) has few test samples, so its metrics carry wide uncertainty.
- **No clinician reviewed** any prediction.
- Multi-visit histories are **simulated** — APTOS is cross-sectional, with one image
  per patient and no follow-up.
- Grad-CAM localises **regions**, not individual lesions, and a plausible-looking
  heatmap is not proof the model reasoned correctly.
- Suggested review intervals are illustrative only and are **not clinical advice**.
""")


# ---------------------------------------------------------------------------
def main() -> None:
    cfg = get_config()
    names = class_names(cfg)
    conn = get_connection(str(get_path(cfg, "db_path")))

    st.title("Diabetic Retinopathy Screening Support")
    disclaimer_banner(cfg)

    page = st.sidebar.radio("Page", ["Screen", "Patient history", "Database", "About"])
    st.sidebar.divider()
    st.sidebar.caption(
        f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n\n"
        f"Seed: {cfg['project']['seed']}"
    )
    st.sidebar.error("Not a medical device. Academic prototype only.")

    {"Screen": page_screen, "Patient history": page_history,
     "Database": page_database, "About": page_about}[page](cfg, conn, names)


if __name__ == "__main__":
    main()
