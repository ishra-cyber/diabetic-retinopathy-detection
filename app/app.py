"""
Retinal Screening Support - Streamlit dashboard.
File location: <project_root>/app/app.py

Run from the project root:
    streamlit run app/app.py

Four pages:
    New screening     upload a fundus image -> stage, confidence, Grad-CAM -> keep on file
    Patient timeline  progression summary, severity chart, visit history, PDF report
    Records           database statistics, stage distribution, activity table, CSV export
    About             grading scale, the active model's test metrics, every trained run

Inference goes through `preprocess_from_config` and `cam_for_image` - the same
functions training and evaluation used - so the app predicts exactly what the
evaluation measured. All storage goes through `src.db.dao`; nothing here writes SQL.
"""

from __future__ import annotations

import hashlib
import html as htmllib
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
import plotly.graph_objects as go
import streamlit as st
import torch

from src.analysis.progression import (
    analyse_progression,
    progression_summary_text,
    simulate_visit_history,
)
from src.data.preprocess import bgr_to_rgb, preprocess_from_config
from src.data.transforms import build_transforms
from src.db import dao
from src.explain.gradcam import cam_for_image
from src.models.evaluate import find_runs, load_checkpoint
from src.models.thresholds import apply_thresholds, scores_to_probabilities
from src.utils.config import class_names, get_path, load_config

st.set_page_config(
    page_title="Retinal Screening Support",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The model the report presents as final. Selected by default when present.
REPORTED_RUN = "densenet121_weighted"

# ───────────────────────────────────────────────────────────── constants
STAGES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
STAGE_COLOR = {0: "#10B981", 1: "#3B82F6", 2: "#F59E0B", 3: "#F97316", 4: "#EF4444"}
STAGE_TINT = {0: "#ECFDF5", 1: "#EFF6FF", 2: "#FFFBEB", 3: "#FFF7ED", 4: "#FEF2F2"}

# One plain-English line per stage, shown under the predicted stage.
STAGE_MEANING = {
    0: "No signs of diabetic eye disease in this photograph.",
    1: "A few tiny bulges in the smallest vessels. Usually watched, not treated.",
    2: "More vessels damaged, some may leak. Usually seen by a specialist within months.",
    3: "Widespread vessel blockage. Usually warrants a prompt specialist appointment.",
    4: "Fragile new vessels are growing. Needs urgent specialist assessment.",
}
FINDINGS = {
    0: "No abnormalities.",
    1: "Microaneurysms only.",
    2: "More than microaneurysms but less than severe NPDR.",
    3: "Any of: &gt;20 intraretinal haemorrhages in each of 4 quadrants, venous beading in "
       "2+ quadrants, or IRMA in 1+ quadrant &mdash; with no signs of proliferation.",
    4: "Neovascularisation, or vitreous / preretinal haemorrhage.",
}
PAGES = ["New screening", "Patient timeline", "Records", "About"]
PLOT_CONFIG = {"displayModeBar": False}
esc = htmllib.escape


# ───────────────────────────────────────────────────────────── styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
  --rs-bg: #F8FAFC; --rs-card: #FFFFFF; --rs-border: #E2E8F0;
  --rs-ink: #0F172A; --rs-body: #334155; --rs-muted: #64748B;
  --rs-mint: #10B981; --rs-mint-dark: #047857; --rs-mint-soft: #ECFDF5;
  --rs-sky: #3B82F6;
  --rs-shadow: 0 1px 3px rgba(15,23,42,0.04), 0 1px 2px rgba(15,23,42,0.03);
  --rs-radius: 14px;
}
.stApp { background: var(--rs-bg); color: var(--rs-ink);
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button { font-family: inherit; }

/* Streamlit chrome */
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
#MainMenu, footer, [data-testid="stSidebarCollapseButton"] { display: none !important; }
.block-container, [data-testid="stMainBlockContainer"] {
  padding-top: 2.25rem !important; padding-bottom: 3rem !important; max-width: 1240px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--rs-border); }
.brand { display: flex; align-items: center; gap: 10px; padding: 0 4px 14px; }
.brand-mark { width: 36px; height: 36px; border-radius: 10px; flex: none;
  background: linear-gradient(135deg, #10B981, #3B82F6);
  display: flex; align-items: center; justify-content: center; }
.brand-name { font-size: 15px; font-weight: 700; color: var(--rs-ink); line-height: 1.15; }
.brand-sub { font-size: 12px; color: var(--rs-muted); }
.nav-label { font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  color: #94A3B8; margin: 18px 4px 6px; }
section[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; width: 100%; display: flex; flex-direction: column; }
section[data-testid="stSidebar"] [data-testid="stRadio"], section[data-testid="stSidebar"] [data-testid="stRadio"] > div { width: 100%; }
section[data-testid="stSidebar"] [role="radiogroup"] label {
  width: 100% !important; display: flex !important; box-sizing: border-box; padding: 9px 12px; border-radius: 10px; margin: 0 !important; cursor: pointer; }
/* hide only the radio circle: any wrapper that holds no text, at either nesting depth */
section[data-testid="stSidebar"] [role="radiogroup"] label > div:not(:has(p)),
section[data-testid="stSidebar"] [role="radiogroup"] label > div > div:not(:has(p)):not(:is(p)) { display: none !important; }
section[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #F1F5F9; }
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked),
section[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] {
  background: var(--rs-mint-soft); box-shadow: inset 3px 0 0 var(--rs-mint); }
section[data-testid="stSidebar"] [role="radiogroup"] label p {
  font-size: 14px; font-weight: 500; color: var(--rs-body) !important; margin: 0; }
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p,
section[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] p {
  color: var(--rs-mint-dark); font-weight: 600; }
.status { margin-top: 28px; background: #F8FAFC; border: 1px solid var(--rs-border);
  border-radius: 12px; padding: 12px 14px; }
.status-title { font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  color: #94A3B8; margin-bottom: 6px; }
.status-row { display: flex; align-items: center; gap: 8px; font-size: 12.5px;
  color: var(--rs-body); padding: 3px 0; }
.dot { width: 7px; height: 7px; border-radius: 50%; background: #94A3B8; flex: none; }
.dot.on { background: var(--rs-mint); box-shadow: 0 0 0 3px rgba(16,185,129,.15); }
.dot.warn { background: #F59E0B; box-shadow: 0 0 0 3px rgba(245,158,11,.15); }

/* Cards: any container created with key="card_..." */
[class*="st-key-card"] { background: var(--rs-card); border: 1px solid var(--rs-border);
  border-radius: var(--rs-radius); box-shadow: var(--rs-shadow); padding: 20px 22px; }

/* Page text */
.rs-head { margin-bottom: 1.25rem; }
.rs-title { font-size: 26px; font-weight: 700; letter-spacing: -0.02em; color: var(--rs-ink); line-height: 1.2; }
.rs-sub { font-size: 14px; color: var(--rs-muted); margin-top: 4px; }
.card-title { font-size: 15px; font-weight: 600; color: var(--rs-ink); }
.card-sub { font-size: 13px; color: var(--rs-muted); margin-top: 2px; }
.section-label { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .06em;
  color: var(--rs-muted); margin: 16px 0 2px; }

/* Metric cards */
.metric { background: #FFFFFF; border: 1px solid var(--rs-border); border-radius: var(--rs-radius);
  box-shadow: var(--rs-shadow); padding: 16px 18px; height: 100%; }
.metric-label { font-size: 13px; color: var(--rs-muted); font-weight: 500; }
.metric-value { font-size: 26px; font-weight: 700; color: var(--rs-ink); letter-spacing: -0.02em;
  margin-top: 6px; font-variant-numeric: tabular-nums; }
.metric-sub { font-size: 12.5px; color: var(--rs-muted); margin-top: 2px; }

/* Badges */
.badge { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
  font-size: 12.5px; font-weight: 600; white-space: nowrap; }
.badge i { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
.src { font-size: 12px; font-weight: 500; padding: 2px 8px; border-radius: 6px; }
.src-real { background: #EFF6FF; color: #1D4ED8; }
.src-simulated { background: #F1F5F9; color: #64748B; }

/* Summary banner */
.banner { background: linear-gradient(90deg, #FFFFFF 0%, #F8FAFC 100%);
  border: 1px solid var(--rs-border); border-left: 4px solid var(--rs-mint);
  border-radius: var(--rs-radius); padding: 16px 20px; box-shadow: var(--rs-shadow); margin: 4px 0 16px; }
.banner-title { font-size: 18px; font-weight: 600; color: var(--rs-ink); }
.banner-sub { font-size: 13px; color: var(--rs-muted); margin-top: 3px; }

/* Result callout */
.result-row { display: flex; gap: 12px; margin-top: 14px; flex-wrap: wrap; }
.callout { flex: 2 1 260px; min-width: 0; border-radius: 12px; border: 1px solid var(--rs-border);
  border-left: 4px solid; padding: 12px 16px; }
.callout-label { font-size: 12px; color: var(--rs-muted); font-weight: 500;
  text-transform: uppercase; letter-spacing: .05em; }
.callout-value { font-size: 20px; font-weight: 700; margin-top: 4px; letter-spacing: -0.01em; }
.conf { flex: 1 1 170px; border: 1px solid var(--rs-border); border-radius: 12px; padding: 12px 16px; min-width: 0; }
.conf-badge { display: inline-block; margin-top: 6px; background: #EFF6FF; color: #1D4ED8;
  font-weight: 700; font-size: 18px; padding: 2px 10px; border-radius: 8px;
  font-variant-numeric: tabular-nums; }

.callout-text { font-size: 12.5px; color: var(--rs-body); margin-top: 6px; line-height: 1.45; }
.note.warn { color: #B45309; }
.stDownloadButton button { border-radius: 10px !important; font-weight: 600 !important; }
/* Empty state & notes */
.empty { text-align: center; padding: 72px 24px; color: var(--rs-muted); }
.empty-icon { width: 52px; height: 52px; border-radius: 14px; background: var(--rs-mint-soft);
  display: flex; align-items: center; justify-content: center; margin: 0 auto 14px; }
.empty-title { font-size: 15px; font-weight: 600; color: var(--rs-ink); }
.empty-sub { font-size: 13px; margin-top: 4px; }
.note { font-size: 13px; margin-top: 8px; }
.note.ok { color: var(--rs-mint-dark); }
.note.muted { color: var(--rs-muted); }

/* Tables */
.table-wrap { max-height: 520px; overflow-y: auto; }
.stMarkdown table.rs-table { width: 100%; border-collapse: collapse; font-size: 13.5px; display: table; }
.stMarkdown table.rs-table tr { background: transparent !important; border: none !important; }
.stMarkdown table.rs-table th { text-align: left; font-size: 12px; font-weight: 600; color: var(--rs-muted);
  text-transform: uppercase; letter-spacing: .05em; padding: 10px 12px; background: #FFFFFF;
  border: none !important; border-bottom: 1px solid var(--rs-border) !important; position: sticky; top: 0; }
.stMarkdown table.rs-table td { padding: 11px 12px; color: var(--rs-body); border: none !important; }
.stMarkdown table.rs-table tbody tr:hover td { background: #F8FAFC; }
.stMarkdown table.rs-table .num { text-align: right; font-variant-numeric: tabular-nums; }
.stMarkdown table.rs-table .strong { color: var(--rs-ink); font-weight: 600; }
.stMarkdown table.rs-table .muted { color: var(--rs-muted); }

/* Key-value list */
.kv { display: flex; justify-content: space-between; gap: 16px; padding: 10px 0;
  border-bottom: 1px solid #F1F5F9; font-size: 13.5px; }
.kv:last-child { border-bottom: none; }
.kv span:first-child { color: var(--rs-muted); }
.kv span:last-child { color: var(--rs-ink); font-weight: 600; text-align: right; font-variant-numeric: tabular-nums; }

/* Widgets — forced light so a leftover dark theme in config.toml can't leak through */
section[data-testid="stSidebar"] { color: var(--rs-ink); }
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div,
[data-baseweb="input"] input, [data-baseweb="base-input"] input, [data-baseweb="textarea"] textarea {
  background: #F1F5F9 !important; color: var(--rs-ink) !important; border-color: transparent !important; }
[data-baseweb="select"] div, [data-baseweb="select"] span, [data-baseweb="select"] svg { color: var(--rs-ink) !important; fill: var(--rs-ink); }
input::placeholder { color: #94A3B8 !important; -webkit-text-fill-color: #94A3B8 !important; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li, [data-baseweb="menu"] { background: #FFFFFF !important; color: var(--rs-ink) !important; }
[data-baseweb="popover"] li:hover { background: #F1F5F9 !important; }
[data-testid="stTextInputRootElement"], [data-testid="stDateInputField"],
[data-testid="stSelectbox"] [role="group"], [data-testid="stNumberInputContainer"] {
  background: #F1F5F9 !important; border-color: transparent !important; border-radius: 10px !important; }
[data-testid="stTextInputRootElement"] input, [data-testid="stDateInputField"] input,
[data-testid="stSelectbox"] input, [data-testid="stDateInput"] input {
  background: transparent !important; color: var(--rs-ink) !important; -webkit-text-fill-color: var(--rs-ink); }
[data-testid="stSelectbox"] button svg { fill: var(--rs-muted) !important; color: var(--rs-muted) !important; }
[data-testid="stDateInputField"] *, [data-testid="stDateInput"] [role="spinbutton"] { color: var(--rs-ink) !important; }
[data-testid="stFileUploaderFile"] button svg, [data-testid="stFileUploaderDeleteBtn"] svg { color: var(--rs-muted) !important; fill: var(--rs-muted) !important; }
[data-testid="stFileUploaderDropzone"] div, [data-testid="stFileUploaderDropzone"] section,
[data-testid="stFileUploaderDropzone"] li, [data-testid="stFileUploaderFile"] {
  background: transparent !important; color: var(--rs-ink) !important; }
[data-testid="stFileUploaderFile"] small, [data-testid="stFileUploaderFile"] span { color: var(--rs-muted) !important; }
[data-testid="stCheckbox"] label > div:has(svg) { background-color: #FFFFFF; border: 1.5px solid #CBD5E1 !important; }
[data-testid="stCheckbox"] label:has(input:checked) > div:has(svg) {
  background-color: var(--rs-mint) !important; border-color: var(--rs-mint) !important; }
[data-testid="stCheckbox"] p, [data-testid="stRadio"] p { color: var(--rs-body) !important; }

[data-testid="stWidgetLabel"] p { font-size: 13px !important; font-weight: 500; color: var(--rs-body); }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"] { border-radius: 10px !important; }
[data-testid="stFileUploaderDropzone"] { background: #F8FAFC; border: 1.5px dashed #CBD5E1; border-radius: 12px; }
button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: var(--rs-mint) !important; border: 1px solid var(--rs-mint) !important;
  color: #FFFFFF !important; border-radius: 10px !important; font-weight: 600 !important; }
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
  background: #059669 !important; border-color: #059669 !important; }
.st-key-card_input .stButton button { width: 100%; height: 44px; }
.st-key-card_result img { border-radius: 12px; width: 100% !important; height: auto; }

/* Tabs as a segmented control (covers older baseweb and newer react-aria markup) */
.stTabs [role="tablist"] { gap: 4px; background: #F1F5F9; padding: 4px; border: none !important;
  border-radius: 10px; width: fit-content; }
.stTabs [role="tab"] { height: 34px; padding: 0 14px; border-radius: 8px; background: transparent;
  border: none !important; display: flex; align-items: center; }
.stTabs [role="tab"] p { font-size: 13px; font-weight: 600; color: var(--rs-muted); }
.stTabs [role="tab"][aria-selected="true"] { background: #FFFFFF; box-shadow: 0 1px 2px rgba(15,23,42,.08); }
.stTabs [role="tab"][aria-selected="true"] p { color: var(--rs-ink); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"],
.stTabs .react-aria-SelectionIndicator { display: none !important; }
.stTabs [role="tabpanel"] { padding-top: 14px; }
</style>
"""


def md(markup: str) -> None:
    """Render raw HTML. Lines are stripped so Markdown never reads indentation as a code block."""
    lines = [line.strip() for line in markup.strip().splitlines() if line.strip()]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


def page_header(title: str, sub: str) -> None:
    md(f'<div class="rs-head"><div class="rs-title">{title}</div><div class="rs-sub">{sub}</div></div>')


def card_title(title: str, sub: str = "") -> None:
    sub_html = f'<div class="card-sub">{sub}</div>' if sub else ""
    md(f'<div class="card-title">{title}</div>{sub_html}')


def spacer(px: int = 16) -> None:
    md(f'<div style="height:{px}px"></div>')


def metric_card(label: str, value: str, sub: str = "", value_color: str | None = None) -> None:
    style = f' style="color:{value_color}"' if value_color else ""
    md(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value"{style}>{value}</div>'
        f'<div class="metric-sub">{sub or "&nbsp;"}</div></div>'
    )


def stage_badge(stage: int) -> str:
    stage = int(stage)
    return (
        f'<span class="badge" style="background:{STAGE_TINT[stage]};color:{STAGE_COLOR[stage]}">'
        f'<i style="background:{STAGE_COLOR[stage]}"></i>Stage {stage} · {STAGES[stage]}</span>'
    )


def source_pill(is_simulated) -> str:
    return ('<span class="src src-simulated">Simulated</span>' if bool(is_simulated)
            else '<span class="src src-real">Real</span>')


def stage_phrase(stage: int) -> str:
    name = STAGES[int(stage)]
    return name if "DR" in name else name.lower()


def note(text: str, kind: str = "muted") -> None:
    md(f'<div class="note {kind}">{text}</div>')


def confidence_wording(confidence: float) -> str:
    """Cautious wording: Milestone 6b measured this model as overconfident."""
    if confidence >= 0.90:
        return "The model is very sure of this."
    if confidence >= 0.75:
        return "The model is fairly sure of this."
    if confidence >= 0.55:
        return "The model leans this way, but is not certain."
    return "The model is genuinely unsure. Treat this as a maybe."


# ───────────────────────────────────────────────────────────── cached resources
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
    """Load a checkpoint once per session. `_cfg` is not hashed by Streamlit."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = get_path(_cfg, "experiments_dir") / run_name
    model, ckpt = load_checkpoint(run_dir / "best.pt", device,
                                  num_classes=_cfg["dataset"]["num_classes"])
    return model, ckpt, device


def available_runs(cfg) -> list[str]:
    return [d.name for d in find_runs(get_path(cfg, "experiments_dir"))]


def run_metrics(cfg, run_name: str) -> dict | None:
    path = get_path(cfg, "experiments_dir") / run_name / "metrics_test.json"
    return json.loads(path.read_text()) if path.is_file() else None


# ───────────────────────────────────────────────────────────── inference
def run_inference(model, ckpt, image_rgb, cfg, device):
    """One prediction, whichever head the checkpoint was trained with.

    A classification checkpoint emits five logits, so `cam_for_image` is taken at
    face value. An ordinal checkpoint emits a single severity score: the stage
    comes from the thresholds stored in the checkpoint (fitted on validation,
    never refitted here) and the five bars are derived from that score.
    """
    head = str(ckpt.get("head", "classification"))
    num_classes = int(cfg["dataset"]["num_classes"])
    result = cam_for_image(model, image_rgb, cfg, device)

    if head != "ordinal":
        result["head"] = "classification"
        result["score"] = None
        return result

    thresholds = ckpt.get("thresholds")
    if not thresholds:
        raise ValueError(
            f"Run '{ckpt.get('run_name')}' uses the ordinal head but its checkpoint "
            "carries no thresholds, so a score cannot be turned into a stage."
        )

    transform = build_transforms(cfg, train=False)
    with torch.no_grad():
        tensor = transform(image_rgb).unsqueeze(0).to(device)
        score = float(model(tensor).float().cpu().numpy().reshape(-1)[0])

    stage = int(apply_thresholds([score], thresholds)[0])
    probabilities = scores_to_probabilities(
        [score], num_classes=num_classes,
        sigma=float(ckpt.get("ordinal_sigma", 0.5)), thresholds=thresholds,
    )[0]

    result.update(head="ordinal", score=score, thresholds=list(thresholds),
                  predicted=stage, probabilities=probabilities,
                  confidence=float(probabilities[stage]))
    return result


# ───────────────────────────────────────────────────────────── charts
def style_fig(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#475569"),
        showlegend=False,
        hoverlabel=dict(bgcolor="#0F172A", font_color="#FFFFFF", bordercolor="#0F172A"),
    )
    fig.update_xaxes(automargin=True, tickfont=dict(color="#64748B"))
    fig.update_yaxes(automargin=True, tickfont=dict(color="#475569"), ticksuffix="   ")
    return fig


def show_chart(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, config=PLOT_CONFIG, theme=None, key=key)


def probability_chart(probs, predicted: int) -> go.Figure:
    probs = [float(p) for p in probs]
    fig = go.Figure(go.Bar(
        x=[p * 100 for p in probs],
        y=[f"Stage {s} · {STAGES[s]}" for s in STAGES],
        orientation="h",
        marker=dict(color=[STAGE_COLOR[s] if s == predicted else "#CBD5E1" for s in STAGES],
                    cornerradius=6),
        text=[f"{p * 100:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(color="#0F172A", size=12),
        cliponaxis=False,
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 118], visible=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, ticks="")
    fig.update_layout(bargap=0.38)
    return style_fig(fig, 220)


def timeline_chart(visits: pd.DataFrame) -> go.Figure:
    dates = pd.to_datetime(visits["visit_date"])
    stages = visits["predicted_stage"].astype(int)
    sim = visits["is_simulated"].astype(bool)
    fig = go.Figure(go.Scatter(
        x=dates, y=stages,
        mode="lines+markers",
        line=dict(color="#10B981", width=2.5, shape="linear"),
        fill="tozeroy",
        fillcolor="rgba(16,185,129,0.10)",
        marker=dict(size=11, color=[STAGE_COLOR[s] for s in stages],
                    line=dict(color="#FFFFFF", width=2)),
        customdata=[[STAGES[s], "Simulated" if f else "Real"] for s, f in zip(stages, sim)],
        hovertemplate="%{x|%d %b %Y}<br>Stage %{y} · %{customdata[0]}<br>%{customdata[1]}<extra></extra>",
    ))
    fig.update_yaxes(range=[-0.25, 4.45], tickvals=list(STAGES),
                     ticktext=[f"{s} · {n}" for s, n in STAGES.items()],
                     gridcolor="#EEF2F6", zeroline=False)
    fig.update_xaxes(showgrid=False, tickformat="%b %Y", linecolor="#E2E8F0")
    if dates.nunique() == 1:
        # One visit: give the lone point some room instead of a run of identical ticks.
        centre = dates.iloc[0]
        fig.update_xaxes(range=[centre - pd.Timedelta(days=90), centre + pd.Timedelta(days=90)],
                         dtick="M1", tickformat="%b %Y")
    return style_fig(fig, 320)


def distribution_chart(by_stage: dict) -> go.Figure:
    counts = [int(by_stage.get(s, by_stage.get(str(s), 0))) for s in STAGES]
    fig = go.Figure(go.Bar(
        x=counts,
        y=[f"Stage {s} · {STAGES[s]}" for s in STAGES],
        orientation="h",
        marker=dict(color=[STAGE_COLOR[s] for s in STAGES], cornerradius=6),
        text=counts,
        textposition="outside",
        textfont=dict(color="#0F172A"),
        cliponaxis=False,
        hovertemplate="%{y}: %{x} visits<extra></extra>",
    ))
    fig.update_xaxes(gridcolor="#EEF2F6", zeroline=False, range=[0, max(max(counts), 1) * 1.15])
    fig.update_yaxes(autorange="reversed", showgrid=False, ticks="")
    fig.update_layout(bargap=0.35)
    return style_fig(fig, 260)


def visits_table(df: pd.DataFrame, show_patient: bool = True) -> None:
    """Borderless HTML table. `df` needs visit_date, predicted_stage, confidence,
    is_simulated and model_version (plus patient_id when show_patient)."""
    rows = []
    for r in df.itertuples():
        when = pd.to_datetime(r.visit_date)
        patient = f"<td class='strong'>{esc(str(r.patient_id))}</td>" if show_patient else ""
        rows.append(
            f"<tr>{patient}<td>{when:%d %b %Y}</td><td>{stage_badge(r.predicted_stage)}</td>"
            f"<td class='num'>{float(r.confidence) * 100:.1f}%</td>"
            f"<td>{source_pill(r.is_simulated)}</td>"
            f"<td class='muted'>{esc(str(r.model_version or '—'))}</td></tr>"
        )
    patient_head = "<th>Patient</th>" if show_patient else ""
    md(
        '<div class="table-wrap"><table class="rs-table"><thead><tr>'
        f"{patient_head}<th>Visit date</th><th>Stage</th>"
        '<th class="num">Confidence</th><th>Source</th><th>Model</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


# ───────────────────────────────────────────────────────────── page: new screening
EYE_ICON = (
    '<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="#10B981" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>'
)


def empty_state(title: str, sub: str) -> None:
    md(f'<div class="empty"><div class="empty-icon">{EYE_ICON}</div>'
       f'<div class="empty-title">{title}</div><div class="empty-sub">{sub}</div></div>')


def save_visit(cfg, conn, patient_id, visit_date, run_name, processed_rgb, result) -> int:
    """Write the processed photo and heatmap to disk, then record the visit."""
    uploads_dir = get_path(cfg, "uploads_dir")
    gradcam_dir = get_path(cfg, "gradcam_dir")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{patient_id}_{datetime.now():%Y%m%d_%H%M%S}"
    image_path = uploads_dir / f"{stem}.png"
    cam_path = gradcam_dir / f"{stem}_cam.png"
    cv2.imwrite(str(image_path), cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(cam_path), cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR))

    return dao.add_visit(
        conn, patient_id=patient_id, visit_date=visit_date,
        predicted_stage=int(result["predicted"]), confidence=float(result["confidence"]),
        probabilities=[float(p) for p in result["probabilities"]],
        image_path=str(image_path), gradcam_path=str(cam_path),
        model_version=run_name, is_simulated=False,
    )


def page_screen(cfg, conn, run_name):
    page_header("New screening", "Upload a retinal photograph to grade diabetic retinopathy severity.")
    left, right = st.columns([5, 6], gap="large")

    with left:
        with st.container(key="card_input"):
            card_title("Screening details", "Fundus photograph and visit information")
            upload = st.file_uploader("Retinal photograph",
                                      type=["png", "jpg", "jpeg", "tif", "tiff"])
            c1, c2 = st.columns(2)
            patient_id = c1.text_input("Patient reference", placeholder="e.g. P001").strip()
            visit_date = c2.date_input("Visit date", value=date.today(), max_value=date.today())
            keep = st.checkbox("Keep this result on file", value=True)
            run = st.button("Run screening", type="primary", disabled=upload is None)
            note(f"Model: <b>{esc(run_name)}</b> &middot; change it in the sidebar.")

            data = upload.getvalue() if upload is not None else None
            key = (hashlib.sha256(data).hexdigest()[:16], run_name) if data else None
            res = st.session_state.result

            if run and data:
                image_bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    st.session_state.result = res = {"key": key, "error": "That file could not be read as an image."}
                else:
                    try:
                        with st.spinner("Loading the model and reading the image..."):
                            model, ckpt, device = get_model(run_name, cfg)
                            processed = bgr_to_rgb(preprocess_from_config(image_bgr, cfg))
                            result = run_inference(model, ckpt, processed, cfg, device)
                        res = {"key": key, "error": None, "original": bgr_to_rgb(image_bgr),
                               "processed": processed, "result": result,
                               "epoch": ckpt.get("epoch"), "saved": None}
                    except Exception as exc:  # noqa: BLE001
                        res = {"key": key, "error": f"The model could not process that image: {esc(str(exc))}"}
                    st.session_state.result = res

                    if res.get("error") is None and keep and patient_id:
                        # Guard against a double click saving the same visit twice.
                        save_key = (*key, patient_id, str(visit_date))
                        if save_key not in st.session_state.saved_keys:
                            try:
                                visit_id = save_visit(cfg, conn, patient_id, visit_date,
                                                      run_name, processed, result)
                                st.session_state.saved_keys.add(save_key)
                                res["saved"] = f"Saved as visit #{visit_id} for {esc(patient_id)}."
                            except ValueError as exc:
                                res["saved"] = f"Could not save: {esc(str(exc))}"
                        else:
                            res["saved"] = f"Already on file for {esc(patient_id)}."

            if res and res.get("key") == key:
                if res.get("error"):
                    note(res["error"], "warn")
                elif res.get("saved"):
                    note(res["saved"], "ok")
                elif keep and not patient_id:
                    note("Add a patient reference to keep this result on file.")

    with right:
        res = st.session_state.result
        with st.container(key="card_result"):
            if not (res and data and res.get("key") == key and not res.get("error")):
                empty_state("No screening yet",
                            "Upload a photograph and run the screening to see the result here.")
                return

            result = res["result"]
            stage = int(result["predicted"])
            conf = float(result["confidence"])

            t1, t2, t3 = st.tabs(["Fundus image", "Grad-CAM heatmap", "Preprocessed"])
            with t1:
                st.image(res["original"])
            with t2:
                st.image(result["overlay"])
                note("Warm areas most influenced the answer. It shows attention, not diagnosis.")
            with t3:
                st.image(res["processed"])
                h, w = res["processed"].shape[:2]
                note(f"Border cropped and resized to {w}&times;{h}, exactly as in training.")

            md(
                '<div class="result-row">'
                f'<div class="callout" style="border-left-color:{STAGE_COLOR[stage]};background:{STAGE_TINT[stage]}">'
                '<div class="callout-label">Predicted stage</div>'
                f'<div class="callout-value" style="color:{STAGE_COLOR[stage]}">Stage {stage} — {STAGES[stage]}</div>'
                f'<div class="callout-text">{STAGE_MEANING[stage]}</div></div>'
                '<div class="conf"><div class="callout-label">Confidence</div>'
                f'<div class="conf-badge">{conf * 100:.1f}%</div>'
                f'<div class="callout-text">{confidence_wording(conf)}</div></div></div>'
            )

            md('<div class="section-label">Probability by stage</div>')
            show_chart(probability_chart(result["probabilities"], stage), key="prob_chart")
            if result.get("head") == "ordinal":
                note(f"Ordinal model: one severity score of <b>{result['score']:.2f}</b> on the 0–4 "
                     "scale, cut at the thresholds fixed in training. The bars are spread from that score.")

            stats = result["stats"]
            border = float(stats["border_fraction"])
            md('<div class="section-label">Heatmap check</div>')
            md(
                f'<div class="kv"><span>Attention on the image edge</span><span>{border:.0%}</span></div>'
                f'<div class="kv"><span>Area the model focused on</span>'
                f'<span>{float(stats["hot_area_fraction"]):.0%}</span></div>'
                f'<div class="kv"><span>Checkpoint</span><span>{esc(run_name)} · epoch {res["epoch"]}</span></div>'
            )
            if border > 0.35:
                note(f"{border:.0%} of the attention sits on the photograph's rim rather than the "
                     "retina, so this answer deserves extra scepticism.", "warn")


# ───────────────────────────────────────────────────────────── page: patient timeline
def page_timeline(cfg, conn, names):
    page_header("Patient timeline", "Track how retinopathy severity changes across visits.")
    patients = dao.list_patients(conn)

    if patients.empty:
        with st.container(key="card_empty_tl"):
            empty_state("No one on file yet",
                        "Screen an image with “Keep this result on file” ticked, or build a demo history below.")
        spacer()
        simulation_card(conn)
        return

    pick, _ = st.columns([1, 2])
    patient_id = pick.selectbox("Patient", patients["patient_id"].tolist())
    visits = dao.get_visits(conn, patient_id)
    if visits.empty:
        note("This patient has no visits recorded.")
        simulation_card(conn)
        return

    a = analyse_progression(visits)
    latest, trend = int(a["latest_stage"]), a["trend"]
    n_sim = int(visits["is_simulated"].sum())

    if a["n_visits"] >= 2:
        headline = f"Currently stage {latest} — {stage_phrase(latest)}, and {trend}"
    else:
        headline = f"Stage {latest} — {stage_phrase(latest)} at the first visit on file"
    sim_line = (f" · {n_sim} of {len(visits)} visits are simulated" if n_sim else "")
    md(
        f'<div class="banner" style="border-left-color:{STAGE_COLOR[latest]}">'
        f'<div class="banner-title">{headline}</div>'
        f'<div class="banner-sub">{esc(progression_summary_text(a, names).replace("NOTE: this history includes SIMULATED visits for demonstration. ", "").strip())}'
        f'{sim_line}</div></div>'
    )

    direction = {
        "worsening": ("Worsening ↑", "#EF4444"),
        "improving": ("Improving ↓", "#10B981"),
        "stable": ("Stable →", "#64748B"),
    }.get(trend, ("—", "#64748B"))
    change = a["stage_change"]
    slope = a.get("slope_per_year")

    cols = st.columns(4)
    with cols[0]:
        metric_card("Visits", str(a["n_visits"]), f"{len(visits) - n_sim} real · {n_sim} simulated")
    with cols[1]:
        metric_card("Latest stage", f"Stage {latest}", STAGES[latest], STAGE_COLOR[latest])
    with cols[2]:
        metric_card("Direction", direction[0], "first vs latest visit", direction[1])
    with cols[3]:
        metric_card("Change since first", "—" if change is None else f"{change:+d}",
                    f"{slope:+.2f} stages / year" if slope is not None else "needs two visits")

    spacer()
    with st.container(key="card_timeline"):
        card_title("Severity over time", "Predicted stage at each visit")
        show_chart(timeline_chart(visits), key="timeline_chart")

    spacer()
    with st.container(key="card_history"):
        head, btn = st.columns([3, 1])
        with head:
            card_title("Visit history", "Oldest visit last")
        with btn:
            pdf = build_pdf_report(cfg, patient_id, visits, a, names)
            if pdf is not None:
                st.download_button("Download PDF", data=pdf, file_name=f"DR_report_{patient_id}.pdf",
                                   mime="application/pdf", width="stretch")
        visits_table(visits.iloc[::-1], show_patient=False)

    latest_row = visits.iloc[-1]
    photos = [(latest_row.get("image_path"), "Photograph"), (latest_row.get("gradcam_path"), "Grad-CAM")]
    photos = [(p, label) for p, label in photos if p and Path(str(p)).is_file()]
    if photos:
        spacer()
        with st.container(key="card_photos"):
            card_title("Latest visit images", f"{pd.to_datetime(latest_row['visit_date']):%d %b %Y}")
            for col, (path, label) in zip(st.columns(2), photos):
                with col:
                    st.image(str(path))
                    note(label)

    spacer()
    simulation_card(conn)


def simulation_card(conn):
    with st.container(key="card_sim"):
        card_title("Build a demo history",
                   "APTOS photographs each person once, so multi-visit histories are simulated "
                   "and flagged as such everywhere.")
        with st.expander("Create simulated visits"):
            c1, c2, c3, c4 = st.columns(4)
            sim_id = c1.text_input("Reference", value="SIM001")
            n_visits = c2.slider("Visits", 2, 8, 5)
            pattern = c3.selectbox("Pattern", ["worsening", "improving", "stable", "fluctuating"])
            start_stage = c4.slider("Starting stage", 0, 4, 1)
            if st.button("Create history"):
                for record in simulate_visit_history(sim_id, n_visits=n_visits,
                                                     start_stage=start_stage, pattern=pattern):
                    dao.add_visit(conn, model_version="SIMULATED", **record)
                st.rerun()


# ───────────────────────────────────────────────────────────── PDF report
def build_pdf_report(cfg, patient_id, visits, analysis, names):
    """One-page PDF summary. Returns None if reportlab is not installed."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas as pdf_canvas
    except ImportError:
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
        ("Observation period", f"{analysis['days_observed']} days" if analysis["days_observed"] else "-"),
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
            pdf.showPage()
            y = height - 20 * mm
            pdf.setFont("Helvetica", 8)
        pdf.drawString(20 * mm, y, str(row["visit_date"]))
        pdf.drawString(55 * mm, y, f"{row['predicted_stage']} - {names[int(row['predicted_stage'])]}")
        pdf.drawString(110 * mm, y, f"{row['confidence']:.1%}")
        pdf.drawString(145 * mm, y, "SIMULATED" if row["is_simulated"] else "recorded")
        y -= 4.5 * mm

    pdf.setFont("Helvetica-Oblique", 7)
    pdf.drawString(20 * mm, 12 * mm, "Generated by an academic screening-support prototype. "
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
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


# ───────────────────────────────────────────────────────────── page: records
def page_records(cfg, conn):
    page_header("Records", "Every screening on file across all patients.")
    stats = dao.database_stats(conn)

    cols = st.columns(4)
    with cols[0]:
        metric_card("People", str(stats["n_patients"]), "patients on file")
    with cols[1]:
        metric_card("Visits", str(stats["n_visits"]), "screenings on file")
    with cols[2]:
        metric_card("Real screenings", str(stats["n_real_visits"]), "run through the model")
    with cols[3]:
        metric_card("Invented for demo", str(stats["n_simulated_visits"]), "simulated history")

    if not stats["n_visits"]:
        spacer()
        with st.container(key="card_empty_rec"):
            empty_state("Nothing stored yet", "Results you keep on file will appear here.")
        return

    spacer()
    with st.container(key="card_distribution"):
        card_title("Stage distribution", "Number of stored visits at each severity stage")
        show_chart(distribution_chart(stats["visits_by_stage"]), key="distribution_chart")

    spacer()
    with st.container(key="card_activity"):
        head, filt = st.columns([3, 2])
        with head:
            card_title("Activity", "Most recent screenings first")
        with filt:
            show = st.radio("Show", ["All", "Real", "Simulated"], horizontal=True,
                            label_visibility="collapsed")
        df = dao.export_all(conn).sort_values(["visit_date", "visit_id"], ascending=False)
        if show != "All":
            df = df[df["is_simulated"].astype(bool) == (show == "Simulated")]
        if df.empty:
            note("No records match this filter yet.")
        else:
            visits_table(df)
        export = dao.export_all(conn)
        st.download_button("Export all records (CSV)",
                           data=export.to_csv(index=False).encode("utf-8"),
                           file_name="dr_visits_export.csv", mime="text/csv")


# ───────────────────────────────────────────────────────────── page: about
def page_about(cfg, run_name):
    page_header("About", "Grading scale, the active model, and every model trained for this project.")
    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.container(key="card_scale"):
            card_title("Severity scale", "International Clinical Diabetic Retinopathy scale")
            rows = "".join(f"<tr><td>{stage_badge(s)}</td><td>{FINDINGS[s]}</td></tr>" for s in STAGES)
            md('<table class="rs-table"><thead><tr><th>Stage</th><th>Typical findings</th></tr></thead>'
               f"<tbody>{rows}</tbody></table>")

        spacer()
        with st.container(key="card_runs"):
            card_title("All trained models", "Held-out test split, 550 images")
            rows = []
            for name in available_runs(cfg):
                m = run_metrics(cfg, name) or {}
                fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "—"  # noqa: E731
                active = " class='strong'" if name == run_name else ""
                rows.append(f"<tr><td{active}>{esc(name)}</td><td class='muted'>{esc(str(m.get('head', '—')))}</td>"
                            f"<td class='num'>{fmt(m.get('qwk'))}</td><td class='num'>{fmt(m.get('accuracy'))}</td>"
                            f"<td class='num'>{fmt(m.get('balanced_accuracy'))}</td></tr>")
            md('<table class="rs-table"><thead><tr><th>Run</th><th>Head</th><th class="num">QWK</th>'
               '<th class="num">Accuracy</th><th class="num">Bal. acc.</th></tr></thead>'
               f"<tbody>{''.join(rows)}</tbody></table>")

    m = run_metrics(cfg, run_name) or {}
    with right:
        with st.container(key="card_model"):
            card_title("Active model", f"{esc(run_name)} · held-out test split")
            pct = lambda v: f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "—"  # noqa: E731
            num = lambda v, d=4: f"{v:.{d}f}" if isinstance(v, (int, float)) else "—"  # noqa: E731
            size = cfg["preprocessing"]["image_size"]
            spec = [
                ("Backbone", m.get("backbone", "—")),
                ("Head", m.get("head", "—")),
                ("Dataset", "APTOS 2019 (3,662 images)"),
                ("Input", f"{size}×{size} RGB, border-cropped"),
                ("Test images", str(m.get("n_samples", "—"))),
                ("Quadratic weighted kappa", num(m.get("qwk"))),
                ("Accuracy", pct(m.get("accuracy"))),
                ("Balanced accuracy", pct(m.get("balanced_accuracy"))),
                ("Macro F1", num(m.get("f1_macro"), 3)),
                ("Macro ROC-AUC", num(m.get("auc_macro"), 3)),
                ("Checkpoint epoch", str(m.get("checkpoint_epoch", "—"))),
                ("Random seed", str(cfg["project"]["seed"])),
            ]
            md("".join(f'<div class="kv"><span>{k}</span><span>{esc(str(v))}</span></div>' for k, v in spec))

        per_class = m.get("per_class") or []
        if per_class:
            spacer()
            with st.container(key="card_recall"):
                card_title("Recall by stage")
                md("".join(
                    f'<div class="kv"><span>{stage_badge(int(c["label"]))}</span>'
                    f'<span>{float(c["recall"]) * 100:.1f}%</span></div>' for c in per_class
                ))


# ───────────────────────────────────────────────────────────── app shell
def sidebar(cfg) -> tuple[str, str | None]:
    with st.sidebar:
        md(
            '<div class="brand"><div class="brand-mark">'
            '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#FFFFFF" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>'
            '</div><div><div class="brand-name">Retinal Screening</div>'
            '<div class="brand-sub">Support</div></div></div>'
        )
        runs = available_runs(cfg)
        run_name = None
        if runs:
            default = runs.index(REPORTED_RUN) if REPORTED_RUN in runs else 0
            run_name = st.selectbox("Model", runs, index=default, label_visibility="collapsed")

        md('<div class="nav-label">Menu</div>')
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")

        gpu = torch.cuda.is_available()
        md(
            '<div class="status"><div class="status-title">System</div>'
            f'<div class="status-row"><span class="dot {"on" if gpu else ""}"></span>'
            f'{"CUDA GPU enabled" if gpu else "Running on CPU"}</div>'
            f'<div class="status-row"><span class="dot on"></span>Random seed {cfg["project"]["seed"]}</div>'
            f'<div class="status-row"><span class="dot {"on" if run_name else "warn"}"></span>'
            f'{"Model ready" if run_name else "No trained model found"}</div></div>'
        )
    return page, run_name


def main() -> None:
    md(CSS)
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("saved_keys", set())

    cfg = get_config()
    names = class_names(cfg)
    conn = get_connection(str(get_path(cfg, "db_path")))
    page, run_name = sidebar(cfg)

    if page == "New screening":
        if run_name is None:
            page_header("New screening", "Upload a retinal photograph to grade diabetic retinopathy severity.")
            with st.container(key="card_nomodel"):
                empty_state("No trained model found", "Train a model first so experiments/ has a best.pt.")
        else:
            page_screen(cfg, conn, run_name)
    elif page == "Patient timeline":
        page_timeline(cfg, conn, names)
    elif page == "Records":
        page_records(cfg, conn)
    else:
        if run_name is None:
            page_header("About", "No trained model found.")
        else:
            page_about(cfg, run_name)


if __name__ == "__main__":
    main()
