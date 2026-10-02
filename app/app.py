"""
Retinal Screening Support — Streamlit dashboard.

Run from the project root:
    streamlit run app/app.py

Requires: streamlit>=1.42, plotly>=5.19, pandas, numpy, pillow
(torch is optional; it is only used to report GPU status.)
"""
from __future__ import annotations

import hashlib
import html as htmllib
import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Retinal Screening Support",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────────────────────────── constants
STAGES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
STAGE_COLOR = {0: "#10B981", 1: "#3B82F6", 2: "#F59E0B", 3: "#F97316", 4: "#EF4444"}
STAGE_TINT = {0: "#ECFDF5", 1: "#EFF6FF", 2: "#FFFBEB", 3: "#FFF7ED", 4: "#FEF2F2"}
MODELS = [
    "densenet121_weighted",
    "resnet50_weighted",
    "effnetb3_weighted",
    "densenet121_ordinal_aug",
]
PAGES = ["New screening", "Patient timeline", "Records", "About"]
WORKSPACES = ["Main clinic", "Outreach camp", "Teaching hospital"]
SEED = 42

# Shown in the sidebar status box. Change to "Live model" once run_model()
# and gradcam_for() call your trained checkpoint instead of the mocks below.
INFERENCE_MODE = "Mock inference"

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
.callout { flex: 1; min-width: 200px; border-radius: 12px; border: 1px solid var(--rs-border);
  border-left: 4px solid; padding: 12px 16px; }
.callout-label { font-size: 12px; color: var(--rs-muted); font-weight: 500;
  text-transform: uppercase; letter-spacing: .05em; }
.callout-value { font-size: 20px; font-weight: 700; margin-top: 4px; letter-spacing: -0.01em; }
.conf { border: 1px solid var(--rs-border); border-radius: 12px; padding: 12px 16px; min-width: 140px; }
.conf-badge { display: inline-block; margin-top: 6px; background: #EFF6FF; color: #1D4ED8;
  font-weight: 700; font-size: 18px; padding: 2px 10px; border-radius: 8px;
  font-variant-numeric: tabular-nums; }

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


def metric_card(label: str, value: str, sub: str = "", value_color: str | None = None) -> None:
    style = f' style="color:{value_color}"' if value_color else ""
    md(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value"{style}>{value}</div>'
        f'<div class="metric-sub">{sub or "&nbsp;"}</div></div>'
    )


def stage_badge(stage: int) -> str:
    return (
        f'<span class="badge" style="background:{STAGE_TINT[stage]};color:{STAGE_COLOR[stage]}">'
        f'<i style="background:{STAGE_COLOR[stage]}"></i>Stage {stage} · {STAGES[stage]}</span>'
    )


def stage_phrase(stage: int) -> str:
    name = STAGES[stage]
    return name if "DR" in name else name.lower()


# ───────────────────────────────────────────────────────────── system status
@st.cache_resource
def gpu_status() -> tuple[str, bool]:
    try:
        import torch  # noqa: WPS433 (optional dependency)
    except ImportError:
        return "PyTorch not installed", False
    if torch.cuda.is_available():
        return "CUDA GPU enabled", True
    return "Running on CPU", False


# ───────────────────────────────────────────────────────────── inference (MOCK)
def run_model(image_bytes: bytes, run_name: str) -> np.ndarray:
    """Return five stage probabilities.

    MOCK: deterministic fake probabilities derived from the image hash, so the
    same photo always gives the same answer. Replace the body with a call to
    your trained checkpoint (preprocess -> model -> softmax) before a real demo,
    then set INFERENCE_MODE = "Live model".
    """
    seed = int(hashlib.sha256(image_bytes + run_name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    stage = int(rng.choice(5, p=[0.40, 0.15, 0.25, 0.10, 0.10]))
    alpha = np.full(5, 0.5)
    alpha[stage] += 8.0
    for neighbour in (stage - 1, stage + 1):
        if 0 <= neighbour < 5:
            alpha[neighbour] += 1.5
    return rng.dirichlet(alpha)


@st.cache_data(show_spinner=False)
def load_image(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((720, 720))
    return img


@st.cache_data(show_spinner=False)
def gradcam_for(image_bytes: bytes) -> Image.Image:
    """MOCK Grad-CAM: a smooth synthetic heatmap blended over the photo.

    Replace with your real Grad-CAM output (e.g. from scripts/m7_gradcam.py).
    """
    img = load_image(image_bytes)
    base = np.asarray(img, dtype=np.float32) / 255.0
    h, w = base.shape[:2]
    rng = np.random.default_rng(int(hashlib.sha256(image_bytes).hexdigest()[:8], 16))
    yy, xx = np.mgrid[0:h, 0:w]
    heat = np.zeros((h, w), dtype=np.float32)
    for _ in range(3):
        cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
        s = rng.uniform(0.07, 0.16) * min(h, w)
        heat += rng.uniform(0.5, 1.0) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    heat /= heat.max()
    r = np.clip(1.5 - np.abs(4 * heat - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * heat - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * heat - 1), 0, 1)
    alpha = (0.55 * heat)[..., None]
    out = base * (1 - alpha) + np.stack([r, g, b], axis=-1) * alpha
    return Image.fromarray((out * 255).astype(np.uint8))


# ───────────────────────────────────────────────────────────── mock records
@st.cache_data(show_spinner=False)
def seed_records() -> list[dict]:
    """Simulated visit history, marked source='Simulated'. Real screenings are added at runtime."""
    rng = np.random.default_rng(SEED)
    today = date.today()
    rows: list[dict] = []
    for i in range(14):
        patient = f"PT-{1001 + i * 7}"
        visit = today - timedelta(days=int(rng.integers(500, 1100)))
        stage = int(rng.choice(5, p=[0.35, 0.25, 0.25, 0.10, 0.05]))
        for v in range(int(rng.integers(2, 7))):
            if v:
                visit += timedelta(days=int(rng.integers(90, 220)))
                stage = int(np.clip(stage + rng.choice([-1, 0, 0, 0, 1, 1]), 0, 4))
            if visit > today:
                break
            rows.append(
                dict(
                    patient=patient,
                    date=visit,
                    stage=stage,
                    confidence=float(rng.uniform(0.62, 0.96)),
                    source="Simulated",
                    model="densenet121_weighted",
                )
            )
    return rows


def records_df() -> pd.DataFrame:
    df = pd.DataFrame(st.session_state.records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def records_table(df: pd.DataFrame) -> None:
    rows = "".join(
        f"<tr><td class='strong'>{esc(str(r.patient))}</td>"
        f"<td>{r.date:%d %b %Y}</td>"
        f"<td>{stage_badge(int(r.stage))}</td>"
        f"<td class='num'>{r.confidence * 100:.1f}%</td>"
        f"<td><span class='src src-{r.source.lower()}'>{r.source}</span></td>"
        f"<td class='muted'>{esc(r.model)}</td></tr>"
        for r in df.itertuples()
    )
    md(
        '<div class="table-wrap"><table class="rs-table"><thead><tr>'
        "<th>Patient</th><th>Visit date</th><th>Stage</th>"
        '<th class="num">Confidence</th><th>Source</th><th>Model</th>'
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


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


def probability_chart(probs: list[float], predicted: int) -> go.Figure:
    labels = [f"Stage {s} · {STAGES[s]}" for s in STAGES]
    fig = go.Figure(
        go.Bar(
            x=[p * 100 for p in probs],
            y=labels,
            orientation="h",
            marker=dict(
                color=[STAGE_COLOR[s] if s == predicted else "#CBD5E1" for s in STAGES],
                cornerradius=6,
            ),
            text=[f"{p * 100:.1f}%" for p in probs],
            textposition="outside",
            textfont=dict(color="#0F172A", size=12),
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, 118], visible=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, ticks="")
    fig.update_layout(bargap=0.38)
    return style_fig(fig, 220)


def timeline_chart(p: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=p["date"],
            y=p["stage"],
            mode="lines+markers",
            line=dict(color="#10B981", width=2.5, shape="linear"),
            fill="tozeroy",
            fillcolor="rgba(16,185,129,0.10)",
            marker=dict(
                size=11,
                color=[STAGE_COLOR[int(s)] for s in p["stage"]],
                line=dict(color="#FFFFFF", width=2),
            ),
            customdata=[STAGES[int(s)] for s in p["stage"]],
            hovertemplate="%{x|%d %b %Y}<br>Stage %{y} · %{customdata}<extra></extra>",
        )
    )
    fig.update_yaxes(
        range=[-0.25, 4.45],
        tickvals=list(STAGES),
        ticktext=[f"{s} · {n}" for s, n in STAGES.items()],
        gridcolor="#EEF2F6",
        zeroline=False,
    )
    fig.update_xaxes(showgrid=False, tickformat="%b %Y", linecolor="#E2E8F0")
    return style_fig(fig, 320)


def distribution_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["stage"].value_counts().reindex(list(STAGES), fill_value=0)
    fig = go.Figure(
        go.Bar(
            x=counts.values,
            y=[f"Stage {s} · {STAGES[s]}" for s in STAGES],
            orientation="h",
            marker=dict(color=[STAGE_COLOR[s] for s in STAGES], cornerradius=6),
            text=counts.values,
            textposition="outside",
            textfont=dict(color="#0F172A"),
            cliponaxis=False,
            hovertemplate="%{y}: %{x} visits<extra></extra>",
        )
    )
    fig.update_xaxes(gridcolor="#EEF2F6", zeroline=False, range=[0, max(int(counts.max()), 1) * 1.15])
    fig.update_yaxes(autorange="reversed", showgrid=False, ticks="")
    fig.update_layout(bargap=0.35)
    return style_fig(fig, 260)


# ───────────────────────────────────────────────────────────── pages
EMPTY_ICON = (
    '<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="#10B981" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>'
)


def page_new_screening() -> None:
    page_header("New screening", "Upload a retinal photograph to grade diabetic retinopathy severity.")
    left, right = st.columns([5, 6], gap="large")

    with left:
        with st.container(key="card_input"):
            md('<div class="card-title">Screening details</div>'
               '<div class="card-sub">Fundus photograph and visit information</div>')
            upload = st.file_uploader("Retinal photograph", type=["png", "jpg", "jpeg"])
            c1, c2 = st.columns(2)
            model = c1.selectbox("Model", MODELS)
            ref = c2.text_input("Patient reference", placeholder="e.g. PT-1001")
            c3, c4 = st.columns(2)
            visit = c3.date_input("Visit date", value=date.today(), max_value=date.today())
            with c4:
                md('<div style="height:30px"></div>')
                keep = st.checkbox("Keep this result on file", value=True)
            run = st.button("Run screening", type="primary", disabled=upload is None)

            data = upload.getvalue() if upload is not None else None
            fid = hashlib.sha256(data).hexdigest()[:16] if data else None

            if run and data:
                probs = run_model(data, model)
                pred = int(np.argmax(probs))
                saved_to = None
                if keep and ref.strip():
                    st.session_state.records.append(
                        dict(
                            patient=ref.strip(),
                            date=visit,
                            stage=pred,
                            confidence=float(probs[pred]),
                            source="Real",
                            model=model,
                        )
                    )
                    saved_to = ref.strip()
                st.session_state.result = dict(
                    fid=fid,
                    model=model,
                    probs=probs.tolist(),
                    pred=pred,
                    saved_to=saved_to,
                    missing_ref=keep and not ref.strip(),
                )

            res = st.session_state.result
            if res and res["fid"] == fid and res["missing_ref"]:
                md('<div class="note muted">Add a patient reference to keep this result on file.</div>')

    with right:
        res = st.session_state.result
        with st.container(key="card_result"):
            if not (res and data and res["fid"] == fid):
                md(
                    f'<div class="empty"><div class="empty-icon">{EMPTY_ICON}</div>'
                    '<div class="empty-title">No screening yet</div>'
                    '<div class="empty-sub">Upload a photograph and run the screening to see the result here.</div></div>'
                )
                return

            img = load_image(data)
            tab_fundus, tab_cam = st.tabs(["Fundus image", "Grad-CAM heatmap"])
            with tab_fundus:
                st.image(img)
            with tab_cam:
                st.image(gradcam_for(data))

            pred = res["pred"]
            conf = res["probs"][pred] * 100
            md(
                '<div class="result-row">'
                f'<div class="callout" style="border-left-color:{STAGE_COLOR[pred]};background:{STAGE_TINT[pred]}">'
                '<div class="callout-label">Predicted stage</div>'
                f'<div class="callout-value" style="color:{STAGE_COLOR[pred]}">Stage {pred} — {STAGES[pred]}</div></div>'
                '<div class="conf"><div class="callout-label">Confidence</div>'
                f'<div class="conf-badge">{conf:.1f}%</div></div></div>'
            )
            md('<div class="section-label">Probability by stage</div>')
            st.plotly_chart(probability_chart(res["probs"], pred), config=PLOT_CONFIG, theme=None, key="prob_chart")
            if res["saved_to"]:
                md(f'<div class="note ok">Saved to {esc(res["saved_to"])}\'s record.</div>')


def page_patient_timeline() -> None:
    page_header("Patient timeline", "Track how retinopathy severity changes across visits.")
    df = records_df()
    patients = sorted(df["patient"].unique())
    default = int(df["patient"].value_counts().reindex(patients).argmax())
    pick, _ = st.columns([1, 2])
    pid = pick.selectbox("Patient", patients, index=default)

    p = df[df["patient"] == pid].sort_values("date").reset_index(drop=True)
    latest, first = int(p["stage"].iloc[-1]), int(p["stage"].iloc[0])

    if len(p) >= 2:
        diff = latest - int(p["stage"].iloc[-2])
        direction = "worsening" if diff > 0 else "improving" if diff < 0 else "stable"
        headline = f"Currently stage {latest} — {stage_phrase(latest)}, and {direction}"
    else:
        direction = "first visit"
        headline = f"Stage {latest} — {stage_phrase(latest)} at first visit"
    dir_style = {
        "worsening": ("Worsening ↑", "#EF4444"),
        "improving": ("Improving ↓", "#10B981"),
        "stable": ("Stable →", "#64748B"),
        "first visit": ("First visit", "#64748B"),
    }[direction]

    md(
        f'<div class="banner" style="border-left-color:{STAGE_COLOR[latest]}">'
        f'<div class="banner-title">{headline}</div>'
        f'<div class="banner-sub">{len(p)} visit{"s" if len(p) != 1 else ""} between '
        f'{p["date"].iloc[0]:%d %b %Y} and {p["date"].iloc[-1]:%d %b %Y}</div></div>'
    )

    change = latest - first
    cols = st.columns(4)
    with cols[0]:
        metric_card("Visits", str(len(p)), "on record")
    with cols[1]:
        metric_card("Latest stage", f"Stage {latest}", STAGES[latest], STAGE_COLOR[latest])
    with cols[2]:
        metric_card("Direction", dir_style[0], "vs previous visit", dir_style[1])
    with cols[3]:
        metric_card("Change since first", f"{change:+d}", f"stages since {p['date'].iloc[0]:%b %Y}")

    md('<div style="height:16px"></div>')
    with st.container(key="card_timeline"):
        md('<div class="card-title">Severity over time</div>'
           '<div class="card-sub">Predicted stage at each visit</div>')
        st.plotly_chart(timeline_chart(p), config=PLOT_CONFIG, theme=None, key="timeline_chart")

    md('<div style="height:16px"></div>')
    with st.container(key="card_history"):
        md('<div class="card-title">Visit history</div>')
        records_table(p.sort_values("date", ascending=False))


def page_records() -> None:
    page_header("Records", "Every screening on file across all patients.")
    df = records_df()
    real = int((df["source"] == "Real").sum())

    cols = st.columns(4)
    with cols[0]:
        metric_card("People", str(df["patient"].nunique()), "unique patients")
    with cols[1]:
        metric_card("Visits", str(len(df)), "screenings on file")
    with cols[2]:
        metric_card("Real screenings", str(real), "run in this app")
    with cols[3]:
        metric_card("Invented for demo", str(len(df) - real), "simulated history")

    md('<div style="height:16px"></div>')
    with st.container(key="card_distribution"):
        md('<div class="card-title">Stage distribution</div>'
           '<div class="card-sub">Number of visits at each severity stage</div>')
        st.plotly_chart(distribution_chart(df), config=PLOT_CONFIG, theme=None, key="distribution_chart")

    md('<div style="height:16px"></div>')
    with st.container(key="card_activity"):
        head, filt = st.columns([3, 2])
        with head:
            md('<div class="card-title">Activity</div>'
               '<div class="card-sub">Most recent screenings first</div>')
        with filt:
            show = st.radio("Show", ["All", "Real", "Simulated"], horizontal=True, label_visibility="collapsed")
        view = df if show == "All" else df[df["source"] == show]
        if view.empty:
            md('<div class="note muted">No records match this filter yet.</div>')
        else:
            records_table(view.sort_values("date", ascending=False))


def page_about() -> None:
    page_header("About", "Grading scale and the model behind each screening.")
    left, right = st.columns([3, 2], gap="large")

    findings = {
        0: "No abnormalities.",
        1: "Microaneurysms only.",
        2: "More than microaneurysms but less than severe NPDR.",
        3: "Any of: &gt;20 intraretinal haemorrhages in each of 4 quadrants, venous beading in "
           "2+ quadrants, or IRMA in 1+ quadrant — with no signs of proliferation.",
        4: "Neovascularisation, or vitreous / preretinal haemorrhage.",
    }
    with left:
        with st.container(key="card_scale"):
            md('<div class="card-title">Severity scale</div>'
               '<div class="card-sub">International Clinical Diabetic Retinopathy scale</div>')
            rows = "".join(
                f"<tr><td>{stage_badge(s)}</td><td>{findings[s]}</td></tr>" for s in STAGES
            )
            md(
                '<table class="rs-table"><thead><tr><th>Stage</th><th>Typical findings</th></tr></thead>'
                f"<tbody>{rows}</tbody></table>"
            )

    with right:
        with st.container(key="card_model"):
            md('<div class="card-title">Model</div>'
               '<div class="card-sub">densenet121_weighted · held-out test split</div>')
            spec = [
                ("Backbone", "DenseNet121"),
                ("Preprocessing", "Ben Graham"),
                ("Loss", "Class-weighted cross-entropy"),
                ("Test images", "550"),
                ("Quadratic weighted kappa", "0.8896"),
                ("QWK 95% CI", "0.862 – 0.915"),
                ("Accuracy", "79.6%"),
                ("Balanced accuracy", "63.7%"),
                ("Macro F1", "0.641"),
                ("Macro ROC-AUC", "0.931"),
                ("Random seed", str(SEED)),
            ]
            md("".join(f'<div class="kv"><span>{k}</span><span>{v}</span></div>' for k, v in spec))

        md('<div style="height:16px"></div>')
        with st.container(key="card_recall"):
            md('<div class="card-title">Recall by stage</div>')
            recall = {0: 0.971, 1: 0.518, 2: 0.727, 3: 0.379, 4: 0.591}
            md("".join(
                f'<div class="kv"><span>{stage_badge(s)}</span><span>{r * 100:.1f}%</span></div>'
                for s, r in recall.items()
            ))


# ───────────────────────────────────────────────────────────── app shell
def sidebar() -> str:
    with st.sidebar:
        md(
            '<div class="brand"><div class="brand-mark">'
            '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#FFFFFF" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>'
            '</div><div><div class="brand-name">Retinal Screening</div>'
            '<div class="brand-sub">Support</div></div></div>'
        )
        st.selectbox("Workspace", WORKSPACES, label_visibility="collapsed")
        md('<div class="nav-label">Menu</div>')
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")

        gpu_label, gpu_on = gpu_status()
        mode_dot = "warn" if INFERENCE_MODE.lower().startswith("mock") else "on"
        md(
            '<div class="status"><div class="status-title">System</div>'
            f'<div class="status-row"><span class="dot {"on" if gpu_on else ""}"></span>{gpu_label}</div>'
            f'<div class="status-row"><span class="dot on"></span>Random seed {SEED}</div>'
            f'<div class="status-row"><span class="dot {mode_dot}"></span>{INFERENCE_MODE}</div>'
            "</div>"
        )
    return page


def main() -> None:
    md(CSS)
    if "records" not in st.session_state:
        st.session_state.records = [dict(r) for r in seed_records()]
    st.session_state.setdefault("result", None)

    page = sidebar()
    {
        "New screening": page_new_screening,
        "Patient timeline": page_patient_timeline,
        "Records": page_records,
        "About": page_about,
    }[page]()


main()
