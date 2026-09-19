"""
Streamlit web application.
File location: <project_root>/app/app.py

Run:
    streamlit run app/app.py

Four pages:
    New screening    upload a fundus image -> plain-language result + Grad-CAM -> save visit
    Patient timeline visit table, progression chart, trend summary, PDF report
    Records          database statistics, recent visits, CSV export
    About            model provenance, metrics, limitations

Design notes
------------
Inference uses `build_transforms(cfg, train=False)` and `preprocess_from_config`
- the exact same functions training used. That is deliberate: an app with its
own copy of the preprocessing drifts from the trained model and predicts
differently from what your evaluation measured, which is a miserable bug to
find the night before a demo.

The interface is written for someone with no medical background reading it over
your shoulder. Every result leads with a sentence in ordinary English - what was
found, and what a person would do about it - and the numbers a marker wants
(probabilities, heatmap diagnostics, checkpoint provenance) sit directly below
rather than behind a click. Colour never carries meaning on its own: every
severity marker ships with its stage number and its name, because roughly one
man in twelve cannot separate the red from the green.
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

import altair as alt
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
from src.data.transforms import build_transforms
from src.db import dao
from src.explain.gradcam import cam_for_image
from src.models.evaluate import find_runs, load_checkpoint
from src.models.thresholds import apply_thresholds, scores_to_probabilities
from src.utils.config import class_names, get_path, load_config

# ---------------------------------------------------------------------------
st.set_page_config(page_title="Retinal Screening Support",
                   page_icon="◎", layout="wide")

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# Severity uses a STATUS palette (good / warning / serious / critical), not a
# categorical or sequential one - the five stages are states, not series. Status
# colour is never the only signal: the stage number and its name travel with it
# everywhere, which is also what keeps the interface readable for colour-blind
# viewers.
#
# Saturated status colours are used for marks and rules only. Text sits on a
# 14%-tint of the same hue instead, because white on #fab219 measures 1.8:1 -
# unreadable - while near-black ink on the tint measures 12.9:1 or better for
# every stage. Contrast was computed, not eyeballed.
SURFACE = "#fbf8f4"       # warm page plane
CARD = "#ffffff"
INK = "#1b1a17"           # 16.4:1 on SURFACE
INK_SOFT = "#6b6560"      # 5.4:1 on SURFACE
HAIRLINE = "#e8e0d6"
ACCENT = "#b4551f"        # 4.7:1 on SURFACE

STAGE_COLOURS = {0: "#0ca30c", 1: "#fab219", 2: "#ec835a", 3: "#d03b3b", 4: "#8f2323"}
STAGE_TINTS = {0: "#daecd4", 1: "#fbeed5", 2: "#f9e8de", 3: "#f5deda", 4: "#ecdad7"}

# Plain-English translation of each stage. `headline` is what a non-medical
# reader sees first; `meaning` explains the finding without jargon; `action` is
# framed as what a person would do, never as an instruction from the software.
STAGE_INFO = {
    0: {
        "short": "No signs found",
        "headline": "No signs of diabetic eye disease were found",
        "meaning": "The blood vessels at the back of this eye look unremarkable in "
                   "this photograph. Nothing in the image suggests damage from diabetes.",
        "action": "Routine yearly screening is the usual advice for anyone living "
                  "with diabetes, even when a check comes back clear.",
        "urgency": "Routine",
    },
    1: {
        "short": "Mild",
        "headline": "Mild early changes were found",
        "meaning": "There are a few tiny bulges in the smallest blood vessels of the "
                   "retina. This is the earliest visible stage and very common. Sight "
                   "is usually unaffected at this point.",
        "action": "This is normally watched rather than treated. A check-up in "
                  "around a year is typical.",
        "urgency": "Watch",
    },
    2: {
        "short": "Moderate",
        "headline": "Moderate changes were found",
        "meaning": "More of the small blood vessels show damage, and some may be "
                   "leaking. Vision is often still normal, but the eye is changing "
                   "more than it should be.",
        "action": "An eye specialist would usually want to see this within a few "
                  "months rather than wait a full year.",
        "urgency": "See a specialist",
    },
    3: {
        "short": "Severe",
        "headline": "Severe changes were found",
        "meaning": "Widespread blockage of the small blood vessels. Parts of the "
                   "retina are not getting enough blood, which is what drives the "
                   "eye toward the sight-threatening stage.",
        "action": "This warrants a prompt appointment with an eye specialist - "
                  "weeks, not months.",
        "urgency": "Prompt referral",
    },
    4: {
        "short": "Proliferative",
        "headline": "Advanced, sight-threatening changes were found",
        "meaning": "The retina has started growing fragile new blood vessels to make "
                   "up for the blocked ones. These bleed easily and are the stage most "
                   "associated with sight loss - though treatment at this point is "
                   "often effective.",
        "action": "This needs urgent specialist assessment. Treatment works best "
                  "when it starts early.",
        "urgency": "Urgent referral",
    },
}


def inject_css() -> None:
    """Additive styling only.

    Colour, widget styling and the sidebar come from .streamlit/config.toml -
    Streamlit generates those from the theme, and CSS that overrides the page
    background while the theme stays dark produces a cream page with black input
    boxes on it. So this stylesheet adds only what Streamlit has no concept of:
    the verdict card, the severity strip, the small key/value rows, and a serif
    face for headings. It sets no text colours and no widget colours.
    """
    st.markdown(
        """
        <style>
          .block-container { padding-top: 2.4rem; max-width: 1180px; }

          h1, h2, h3, .dr-serif {
            font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
            letter-spacing: -0.01em;
          }
          h1 { font-size: 2.0rem !important; font-weight: 600 !important; }
          h2 { font-size: 1.32rem !important; font-weight: 600 !important;
               margin-top: 1.7rem !important; }
          h3 { font-size: 1.06rem !important; font-weight: 600 !important; }

          .dr-masthead { border-bottom: 1px solid HAIRLINE;
                         padding-bottom: .65rem; margin-bottom: 1.1rem; }
          .dr-masthead .sub { color: INK_SOFT; font-size: .92rem; }

          .dr-card { background: #ffffff; border: 1px solid HAIRLINE;
                     border-radius: 14px; padding: 1.1rem 1.25rem;
                     margin-bottom: .9rem; height: 100%; }

          .dr-verdict { border-radius: 16px; padding: 1.35rem 1.55rem;
                        margin: .4rem 0 1.1rem 0; }
          .dr-verdict .stage { font-size: .78rem; letter-spacing: .09em;
                               text-transform: uppercase; color: INK_SOFT; }
          .dr-verdict .head { font-family: Georgia, "Iowan Old Style", serif;
                              font-size: 1.5rem; line-height: 1.3;
                              margin: .35rem 0 .5rem 0; color: INK; }
          .dr-verdict .body { font-size: 1.0rem; line-height: 1.6;
                              max-width: 62ch; color: INK; }

          .dr-scale { display: flex; gap: 6px; margin: .2rem 0 .3rem 0; }
          .dr-step { flex: 1; border-radius: 7px; padding: .5rem .35rem .45rem .35rem;
                     text-align: center; font-size: .73rem; line-height: 1.25;
                     border: 1px solid transparent; }
          .dr-step .n { display: block; font-weight: 700; font-size: .95rem; }

          .dr-note { border-left: 3px solid ACCENT; padding: .5rem 0 .5rem .85rem;
                     color: INK_SOFT; font-size: .9rem; line-height: 1.55; }

          .dr-kv { display: flex; justify-content: space-between; gap: 1rem;
                   padding: .4rem 0; border-bottom: 1px dotted HAIRLINE;
                   font-size: .92rem; }
          .dr-kv .k { color: INK_SOFT; }
          .dr-kv .v { font-variant-numeric: tabular-nums; }

          .dr-disclaimer { background: #fdf3ef; border: 1px solid #f0d5c9;
                           border-radius: 10px; padding: .7rem .95rem;
                           font-size: .84rem; line-height: 1.5; color: #5d4034; }

          .stButton > button, .stDownloadButton > button {
            border-radius: 9px; font-weight: 600; padding: .45rem 1.1rem;
          }
          div[data-testid="stImage"] img { border-radius: 10px; }
        </style>
        """.replace("HAIRLINE", HAIRLINE)
           .replace("INK_SOFT", INK_SOFT)
           .replace("ACCENT", ACCENT)
           .replace("INK", INK),
        unsafe_allow_html=True,
    )


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
# Shared UI pieces
# ---------------------------------------------------------------------------
def masthead(title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='dr-masthead'>"
        f"<h1 style='margin:0 0 .15rem 0'>{title}</h1>"
        f"<div class='sub'>{subtitle}</div></div>",
        unsafe_allow_html=True,
    )


def disclaimer_banner(cfg):
    st.markdown(
        "<div class='dr-disclaimer'><b>Academic prototype - not a medical device.</b> "
        + " ".join(cfg["app"]["disclaimer"].split()).replace(
            "ACADEMIC PROTOTYPE - NOT A MEDICAL DEVICE. ", "")
        + "</div>",
        unsafe_allow_html=True,
    )


def confidence_wording(confidence: float) -> str:
    """Turn a softmax number into a phrase a non-specialist can act on.

    The bands are deliberately cautious. Milestone 6b measured this model to be
    overconfident - mean confidence 0.83 against 0.80 accuracy, with fitted
    temperatures above 1 for every architecture - so a raw 0.95 does not mean
    95% of such calls are right, and the wording should not imply it does.
    """
    if confidence >= 0.90:
        return "the model is very sure of this"
    if confidence >= 0.75:
        return "the model is fairly sure of this"
    if confidence >= 0.55:
        return "the model leans this way, but is not certain"
    return "the model is genuinely unsure - treat this as a maybe"


def severity_scale(stage: int, names) -> None:
    """The five stages as a strip, with the current one raised.

    A person who has never seen a grading scale learns the whole thing in one
    glance: where this result sits, and what is on either side of it.
    """
    cells = []
    for k in range(len(names)):
        current = (k == int(stage))
        bg = STAGE_TINTS[k] if current else "#f4f1ec"
        border = STAGE_COLOURS[k] if current else "transparent"
        weight = "700" if current else "500"
        colour = INK if current else INK_SOFT
        cells.append(
            f"<div class='dr-step' style='background:{bg};border-color:{border};"
            f"color:{colour};font-weight:{weight}'>"
            f"<span class='n'>{k}</span>{STAGE_INFO[k]['short']}</div>"
        )
    st.markdown("<div class='dr-scale'>" + "".join(cells) + "</div>",
                unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:.78rem;color:{INK_SOFT};margin-bottom:1rem'>"
        f"Stage 0 on the left is a healthy retina; stage 4 on the right is the "
        f"sight-threatening end of the scale. This image was placed at "
        f"<b>stage {int(stage)}</b>.</div>",
        unsafe_allow_html=True,
    )


def verdict_card(stage: int, confidence: float) -> None:
    info = STAGE_INFO[int(stage)]
    st.markdown(
        f"<div class='dr-verdict' style='background:{STAGE_TINTS[int(stage)]};"
        f"border-left:5px solid {STAGE_COLOURS[int(stage)]}'>"
        f"<div class='stage'>Stage {int(stage)} &nbsp;&middot;&nbsp; {info['urgency']}"
        f" &nbsp;&middot;&nbsp; {confidence:.0%} confidence</div>"
        f"<div class='head'>{info['headline']}</div>"
        f"<div class='body'>{info['meaning']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def probability_chart(probabilities, names):
    """Horizontal bars, one series, one hue.

    The bars encode a single magnitude - how much of the model's belief landed on
    each stage - so they take one colour, not five. Severity colour would be a
    second meaning on the same mark and would imply the chart is about severity
    rather than certainty. The predicted stage is distinguished by ink weight and
    a direct label instead.
    """
    frame = pd.DataFrame({
        "stage": [f"{i} - {n}" for i, n in enumerate(names)],
        "probability": [float(p) for p in probabilities],
    })
    top = int(np.argmax(frame["probability"].to_numpy()))
    frame["is_top"] = [i == top for i in range(len(frame))]

    base = alt.Chart(frame).encode(
        y=alt.Y("stage:N", sort=None, title=None,
                axis=alt.Axis(labelColor=INK, labelFontSize=12, domain=False,
                              ticks=False, labelPadding=8)),
        x=alt.X("probability:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", grid=True, gridColor=HAIRLINE,
                              tickCount=5, labelColor=INK_SOFT, domain=False,
                              ticks=False)),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, size=18).encode(
        color=alt.condition(alt.datum.is_top, alt.value(ACCENT), alt.value("#ded5c8")),
        tooltip=[alt.Tooltip("stage:N", title="Stage"),
                 alt.Tooltip("probability:Q", title="Share of belief", format=".1%")],
    )
    labels = base.mark_text(align="left", dx=6, fontSize=12, color=INK).encode(
        text=alt.Text("probability:Q", format=".0%"),
        opacity=alt.condition(alt.datum.probability > 0.04, alt.value(1), alt.value(0)),
    )
    return (bars + labels).properties(height=28 * len(names) + 20).configure_view(
        strokeWidth=0).configure(background="transparent")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(model, ckpt, image_rgb, cfg, device):
    """One prediction, whichever head the checkpoint was trained with.

    This exists because the two heads return different shapes and the app must
    not care. A classification checkpoint emits five logits and
    ``cam_for_image`` can be taken at face value. An ordinal checkpoint emits a
    SINGLE number - the severity score - so its softmax is a one-element vector,
    and reading ``probabilities`` from it gave a length-1 array where the rest of
    the page expected five. That mismatch is what raised
    "All arrays must be of the same length" the moment an ordinal run was picked
    in the model dropdown, which the default selection does as soon as an ordinal
    run has the best test QWK.

    For an ordinal checkpoint the stage comes from the thresholds stored in the
    checkpoint - the ones fitted on validation during training, never refitted
    here - and the five-way bars are derived from the score rather than learned.
    That distinction is surfaced in the interface rather than hidden.
    """
    head = str(ckpt.get("head", "classification"))
    num_classes = int(cfg["dataset"]["num_classes"])

    # The overlay and its diagnostics come from Grad-CAM either way. With one
    # output the gradient is taken on the severity score itself, which is the
    # right attribution target.
    result = cam_for_image(model, image_rgb, cfg, device)

    if head != "ordinal":
        result["head"] = "classification"
        result["score"] = None
        return result

    thresholds = ckpt.get("thresholds")
    if not thresholds:
        raise ValueError(
            f"Run '{ckpt.get('run_name')}' uses the ordinal head but its checkpoint "
            "carries no thresholds, so a score cannot be turned into a stage. "
            "Re-run training for it, or pick a different model."
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

    result["head"] = "ordinal"
    result["score"] = score
    result["thresholds"] = list(thresholds)
    result["predicted"] = stage
    result["probabilities"] = probabilities
    result["confidence"] = float(probabilities[stage])
    return result


# ---------------------------------------------------------------------------
# Page: New screening
# ---------------------------------------------------------------------------
def page_screen(cfg, conn, names):
    masthead("New screening", "Upload one retinal photograph")

    runs = available_runs(cfg)
    if not runs:
        st.warning("No trained model found in `experiments/`. Train a model first "
                   "(Milestones 4-5).")
        return

    col_a, col_b = st.columns([2, 1], gap="large")
    with col_a:
        uploaded = st.file_uploader(
            "Retinal photograph",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            help="A fundus photograph - the picture taken by the camera that "
                 "photographs the back of the eye.",
        )
    with col_b:
        default = best_run_name(cfg)
        run_name = st.selectbox("Model", runs,
                                index=runs.index(default) if default in runs else 0)
        patient_id = st.text_input("Patient reference", value="P001",
                                   help="Any label you like. Do not enter real "
                                        "patient details - this is a prototype.")
        visit_date = st.date_input("Visit date", value=date.today())
        save_record = st.checkbox("Keep this result on file", value=True)

    if uploaded is None:
        st.markdown(
            "<div class='dr-note'>Pick an image to begin. Any photograph from "
            "<code>data/raw/train_images/</code> works for a demonstration.<br><br>"
            "<b>What happens next:</b> the picture is cropped and colour-corrected "
            "the same way the training images were, the model places it on a "
            "0-4 scale, and a heat map shows which part of the retina drove that "
            "answer.</div>",
            unsafe_allow_html=True,
        )
        return

    try:
        buffer = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
        image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image_bgr is None:
            st.error("That file could not be read as an image.")
            return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read the upload: {exc}")
        return

    with st.spinner("Loading the model..."):
        try:
            model, ckpt, device = get_model(run_name, cfg)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load the model: {exc}")
            return

    # Exactly the preprocessing used in training - see the module docstring.
    processed_rgb = bgr_to_rgb(preprocess_from_config(image_bgr, cfg))

    with st.spinner("Reading the image..."):
        try:
            result = run_inference(model, ckpt, processed_rgb, cfg, device)
        except Exception as exc:  # noqa: BLE001
            st.error(f"The model could not process that image: {exc}")
            return

    stage = int(result["predicted"])
    confidence = float(result["confidence"])

    # -- the answer, in English ---------------------------------------------
    st.divider()
    verdict_card(stage, confidence)
    severity_scale(stage, names)

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown(
            f"<div class='dr-card'><h3 style='margin-top:0'>What usually happens next</h3>"
            f"<p style='margin:0;line-height:1.6'>{STAGE_INFO[stage]['action']}</p></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"<div class='dr-card'><h3 style='margin-top:0'>How sure is it?</h3>"
            f"<p style='margin:0 0 .5rem 0;line-height:1.6'>Of everything it "
            f"considered, {confidence:.0%} of the model's belief landed on stage "
            f"{stage} - {confidence_wording(confidence)}.</p>"
            f"<p style='margin:0;font-size:.85rem;color:{INK_SOFT};line-height:1.55'>"
            f"This model is known to overstate its own certainty, so read the "
            f"percentage as a rough strength of opinion rather than a probability "
            f"of being right.</p></div>",
            unsafe_allow_html=True,
        )

    # -- the pictures --------------------------------------------------------
    st.subheader("The image, step by step")
    c1, c2, c3 = st.columns(3, gap="medium")
    c1.image(bgr_to_rgb(image_bgr), use_container_width=True)
    c1.markdown(f"<div style='font-size:.82rem;color:{INK_SOFT}'><b>As uploaded.</b> "
                f"The original photograph.</div>", unsafe_allow_html=True)
    c2.image(processed_rgb, use_container_width=True)
    c2.markdown(f"<div style='font-size:.82rem;color:{INK_SOFT}'><b>After tidying.</b> "
                f"Black border trimmed and the image squared to "
                f"{processed_rgb.shape[1]}x{processed_rgb.shape[0]}, so every "
                f"picture reaches the model the same way.</div>",
                unsafe_allow_html=True)
    c3.image(result["overlay"], use_container_width=True)
    c3.markdown(f"<div style='font-size:.82rem;color:{INK_SOFT}'><b>Where it looked.</b> "
                f"Warm areas are the parts of the retina that most influenced the "
                f"answer. It shows attention, not diagnosis.</div>",
                unsafe_allow_html=True)

    # -- the numbers, still on the page --------------------------------------
    st.subheader("The numbers behind the answer")
    n_left, n_right = st.columns([1.25, 1], gap="large")

    with n_left:
        if result.get("head") == "ordinal":
            blurb = (f"This model predicts a single severity score rather than five "
                     f"separate probabilities - it read <b>{result['score']:.2f}</b> on "
                     f"the 0-4 scale, and the cut-points fixed during training put that "
                     f"at stage {stage}. The bars below are spread out from that one "
                     f"number, so read them as nearness to each stage rather than as "
                     f"five independent opinions.")
        else:
            blurb = ("How the model divided its belief across the five stages. A tall "
                     "second bar means two stages looked alike to it.")
        st.markdown(f"<div style='font-size:.9rem;color:{INK_SOFT};margin-bottom:.4rem'>"
                    f"{blurb}</div>", unsafe_allow_html=True)
        st.altair_chart(probability_chart(result["probabilities"], names),
                        use_container_width=True)

    with n_right:
        stats = result["stats"]
        border = float(stats["border_fraction"])
        st.markdown(
            "<div style='font-size:.9rem;color:" + INK_SOFT + ";margin-bottom:.4rem'>"
            "Checks on the heat map itself - a way of catching the model looking at "
            "the wrong thing.</div>", unsafe_allow_html=True)
        rows = [
            ("Attention on the image edge", f"{border:.0%}"),
            ("Area the model focused on", f"{float(stats['hot_area_fraction']):.0%}"),
            ("Strongest response", f"{float(stats['peak_value']):.2f}"),
            ("Average response", f"{float(stats['mean_value']):.2f}"),
        ]
        st.markdown(
            "".join(f"<div class='dr-kv'><span class='k'>{k}</span>"
                    f"<span class='v'>{v}</span></div>" for k, v in rows),
            unsafe_allow_html=True)
        if border > 0.35:
            st.markdown(
                f"<div class='dr-note' style='border-left-color:{STAGE_COLOURS[3]};"
                f"margin-top:.7rem'><b>Worth a second look.</b> {border:.0%} of the "
                f"model's attention sat around the rim of the photograph rather than "
                f"on the retina. That can mean it is keying on the camera edge, so "
                f"this particular answer deserves more scepticism than usual.</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='dr-note' style='margin-top:.7rem'>Attention is "
                f"concentrated on the retina rather than the photograph's edge, "
                f"which is what you want to see.</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div style='font-size:.8rem;color:{INK_SOFT};margin-top:.6rem'>"
        f"Model <code>{ckpt.get('backbone')}</code> from run "
        f"<code>{run_name}</code>, checkpoint epoch {ckpt.get('epoch')}.</div>",
        unsafe_allow_html=True)

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

        st.divider()
        if st.button("Save this visit", type="primary"):
            try:
                visit_id = dao.add_visit(
                    conn, patient_id=patient_id, visit_date=visit_date,
                    predicted_stage=stage, confidence=confidence,
                    probabilities=result["probabilities"],
                    image_path=str(image_path), gradcam_path=str(cam_path),
                    model_version=run_name, is_simulated=False,
                )
                st.success(f"Saved as visit #{visit_id} for {patient_id}. "
                           f"It will appear under Patient timeline.")
            except ValueError as exc:
                st.error(f"Could not save: {exc}")


# ---------------------------------------------------------------------------
# Page: Patient timeline
# ---------------------------------------------------------------------------
def page_history(cfg, conn, names):
    masthead("Patient timeline", "How one person's readings change over time")

    patients = dao.list_patients(conn)
    if patients.empty:
        st.markdown("<div class='dr-note'>No one on file yet. Screen an image on the "
                    "previous page, or build a worked example below.</div>",
                    unsafe_allow_html=True)
        _simulation_controls(conn)
        return

    ids = patients["patient_id"].tolist()
    patient_id = st.selectbox("Patient", ids)

    visits = dao.get_visits(conn, patient_id)
    if visits.empty:
        st.warning("This patient has no visits recorded.")
        _simulation_controls(conn)
        return

    if visits["is_simulated"].any():
        st.markdown(
            f"<div class='dr-disclaimer'><b>{int(visits['is_simulated'].sum())} of "
            f"{len(visits)} of these visits are invented.</b> The APTOS dataset "
            f"photographs each person once, so there is no real follow-up data "
            f"anywhere in this project. These records exist purely to show that the "
            f"tracking works, and they are flagged as synthetic in the database and "
            f"in the PDF.</div>",
            unsafe_allow_html=True)

    analysis = analyse_progression(visits)
    latest = analysis["latest_stage"]

    if latest is not None:
        st.markdown(
            f"<div class='dr-verdict' style='background:{STAGE_TINTS[int(latest)]};"
            f"border-left:5px solid {STAGE_COLOURS[int(latest)]}'>"
            f"<div class='stage'>Most recent reading &nbsp;&middot;&nbsp; "
            f"{analysis['n_visits']} visits on file</div>"
            f"<div class='head'>Currently stage {int(latest)} - "
            f"{STAGE_INFO[int(latest)]['short'].lower()}, and "
            f"{analysis['trend']}</div>"
            f"<div class='body'>{progression_summary_text(analysis, names)}</div>"
            f"</div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Visits", analysis["n_visits"])
    m2.metric("Latest stage",
              f"{latest} - {STAGE_INFO[int(latest)]['short']}" if latest is not None else "-")
    m3.metric("Direction", analysis["trend"].title())
    m4.metric("Change since first visit",
              f"{analysis['stage_change']:+d} stages"
              if analysis["stage_change"] is not None else "-")

    if latest is not None:
        suggestion = next_review_suggestion(latest, analysis["trend"])
        st.markdown(
            f"<div class='dr-note'>For illustration only, a pattern like this is "
            f"often reviewed after about <b>{suggestion['months']} months</b> "
            f"(around {suggestion['approx_date']}). {suggestion['disclaimer']}</div>",
            unsafe_allow_html=True)

    if len(visits) >= 1:
        st.subheader("Readings over time")
        st.pyplot(progression_figure(visits, names, patient_id))

    st.subheader("Every visit on file")
    display = visits[["visit_id", "visit_date", "predicted_stage", "confidence",
                      "model_version", "is_simulated"]].copy()
    display["Stage"] = display["predicted_stage"].map(
        lambda s: f"{int(s)} - {STAGE_INFO[int(s)]['short']}")
    display["Source"] = display["is_simulated"].map(
        lambda flag: "Invented (demo)" if flag else "Recorded")
    display = display.rename(columns={"visit_id": "Visit", "visit_date": "Date",
                                      "confidence": "Confidence",
                                      "model_version": "Model"})
    st.dataframe(display[["Visit", "Date", "Stage", "Confidence", "Model", "Source"]],
                 hide_index=True, use_container_width=True)

    with st.expander("Photographs and heat maps from each visit"):
        for _, row in visits.iterrows():
            cols = st.columns([1, 1, 2])
            for col, key, label in ((cols[0], "image_path", "photograph"),
                                    (cols[1], "gradcam_path", "heat map")):
                path = row.get(key)
                if path and Path(path).is_file():
                    col.image(str(path), caption=f"{row['visit_date']} {label}",
                              use_container_width=True)
                else:
                    col.caption(f"no {label} stored")
            cols[2].markdown(
                f"**Visit {row['visit_id']}** - {row['visit_date']}<br>"
                f"Stage {int(row['predicted_stage'])} - "
                f"{STAGE_INFO[int(row['predicted_stage'])]['short']}, "
                f"{row['confidence']:.0%} confidence", unsafe_allow_html=True)

    pdf = build_pdf_report(cfg, patient_id, visits, analysis, names)
    if pdf is not None:
        st.download_button("Download a summary PDF", data=pdf,
                           file_name=f"DR_report_{patient_id}.pdf",
                           mime="application/pdf", type="primary")

    st.divider()
    _simulation_controls(conn)


def _simulation_controls(conn):
    with st.expander("Build a worked example (invented data)"):
        st.markdown(
            "<div style='font-size:.88rem;line-height:1.55'>APTOS photographs each "
            "person once, so this project has no real follow-up data. This creates a "
            "made-up sequence of visits so the tracking page has something to show. "
            "Every record it writes is flagged as synthetic wherever it appears.</div>",
            unsafe_allow_html=True)
        st.write("")
        c1, c2, c3, c4 = st.columns(4)
        sim_id = c1.text_input("Reference", value="SIM001")
        n_visits = c2.slider("How many visits", 2, 8, 5)
        pattern = c3.selectbox("Pattern", ["worsening", "improving", "stable", "fluctuating"])
        start_stage = c4.slider("Starting stage", 0, 4, 1)

        if st.button("Create the example"):
            records = simulate_visit_history(sim_id, n_visits=n_visits,
                                             start_stage=start_stage, pattern=pattern)
            for record in records:
                dao.add_visit(conn, model_version="SIMULATED", **record)
            st.success(f"Created {len(records)} invented visits for {sim_id}.")
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
# Page: Records
# ---------------------------------------------------------------------------
def page_database(cfg, conn, names):
    masthead("Records", "Everything this prototype has stored")
    stats = dao.database_stats(conn)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("People", stats["n_patients"])
    c2.metric("Visits", stats["n_visits"])
    c3.metric("Real screenings", stats["n_real_visits"])
    c4.metric("Invented for demo", stats["n_simulated_visits"])

    if stats["n_simulated_visits"]:
        st.markdown(
            f"<div class='dr-disclaimer'>{stats['n_simulated_visits']} of "
            f"{stats['n_visits']} visits were invented to demonstrate the tracking "
            f"page. They are marked as such everywhere they appear.</div>",
            unsafe_allow_html=True)

    if stats["visits_by_stage"]:
        st.subheader("How the stored readings are spread across the scale")
        frame = pd.DataFrame({
            "stage": [f"{int(k)} - {STAGE_INFO[int(k)]['short']}"
                      for k in stats["visits_by_stage"]],
            "count": list(stats["visits_by_stage"].values()),
        })
        chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=4, size=20).encode(
            y=alt.Y("stage:N", sort=None, title=None,
                    axis=alt.Axis(labelColor=INK, domain=False, ticks=False,
                                  labelPadding=8)),
            x=alt.X("count:Q", title=None,
                    axis=alt.Axis(grid=True, gridColor=HAIRLINE, tickMinStep=1,
                                  labelColor=INK_SOFT, domain=False, ticks=False)),
            color=alt.value(ACCENT),
            tooltip=[alt.Tooltip("stage:N", title="Stage"),
                     alt.Tooltip("count:Q", title="Visits")],
        ).properties(height=28 * len(frame) + 20).configure_view(
            strokeWidth=0).configure(background="transparent")
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Most recent activity")
    st.dataframe(dao.recent_visits(conn, 25), hide_index=True, use_container_width=True)

    export = dao.export_all(conn)
    if not export.empty:
        st.download_button("Export everything as a spreadsheet (CSV)",
                           data=export.to_csv(index=False).encode("utf-8"),
                           file_name="dr_visits_export.csv", mime="text/csv")

    st.markdown(
        f"<div style='font-size:.8rem;color:{INK_SOFT};margin-top:1rem'>"
        f"Stored in <code>{get_path(cfg, 'db_path')}</code>, schema "
        f"v{stats['schema_version']}.</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
def page_about(cfg, conn, names):
    masthead("About", "What this is, and what it is not")

    st.markdown(
        "<div class='dr-card'><h3 style='margin-top:0'>In one paragraph</h3>"
        "<p style='margin:0;line-height:1.65;max-width:68ch'>Diabetes can damage the "
        "blood vessels at the back of the eye, and the damage is graded on a "
        "five-point scale from none to sight-threatening. This prototype looks at a "
        "photograph of the retina, places it on that scale, and shows which part of "
        "the picture drove its answer. It then keeps a record so the readings can be "
        "compared over time. It is a student research project, not a product, and no "
        "eye doctor has reviewed anything it produces.</p></div>",
        unsafe_allow_html=True)

    st.subheader("The five stages")
    for k in range(len(names)):
        st.markdown(
            f"<div style='display:flex;gap:1rem;align-items:flex-start;padding:.7rem 0;"
            f"border-bottom:1px solid {HAIRLINE}'>"
            f"<div style='flex:0 0 116px;background:{STAGE_TINTS[k]};"
            f"border-left:4px solid {STAGE_COLOURS[k]};border-radius:6px;"
            f"padding:.4rem .6rem;font-weight:600;font-size:.85rem'>"
            f"{k} &middot; {STAGE_INFO[k]['short']}</div>"
            f"<div style='flex:1;font-size:.92rem;line-height:1.6;max-width:62ch'>"
            f"{STAGE_INFO[k]['meaning']}</div></div>",
            unsafe_allow_html=True)

    st.subheader("How it was built")
    st.markdown(f"""
| | |
|---|---|
| Dataset | APTOS 2019 Blindness Detection - 3,662 labelled photographs from rural India |
| Input | {cfg['preprocessing']['image_size']}x{cfg['preprocessing']['image_size']} RGB, black borders cropped, ImageNet-normalised |
| Headline metric | Quadratic Weighted Kappa - it penalises being three stages out far more than one |
| Explanation method | Grad-CAM on the final convolutional layer |
| Storage | SQLite ({get_path(cfg, 'db_path').name}) |
| Random seed | {cfg['project']['seed']} |
""")

    st.subheader("Models trained for this project")
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

    st.subheader("What it cannot do")
    st.markdown("""
- It was trained and tested on **one dataset** and has never been tried on images
  from a different clinic, camera or country.
- The **Severe** stage has very few examples, so its numbers carry wide uncertainty
  and it is the stage the model gets wrong most often.
- **No eye doctor has checked a single prediction** it has made.
- Multi-visit histories are **invented**. APTOS photographs each person once.
- The heat map shows *where* the model looked, not *what it saw*. A heat map that
  lands on the right area is not proof the reasoning was right.
- Suggested review intervals are illustrative. They are **not medical advice** and
  no decision about a real person should rest on them.
""")


# ---------------------------------------------------------------------------
def main() -> None:
    cfg = get_config()
    names = class_names(cfg)
    conn = get_connection(str(get_path(cfg, "db_path")))

    inject_css()

    with st.sidebar:
        st.markdown(
            f"<div class='dr-serif' style='font-size:1.18rem;line-height:1.3;"
            f"margin-bottom:.15rem'>Retinal Screening Support</div>"
            f"<div style='font-size:.8rem;color:{INK_SOFT};margin-bottom:1.1rem'>"
            f"Diabetic eye disease, graded 0-4</div>", unsafe_allow_html=True)

        page = st.radio("Go to", ["New screening", "Patient timeline",
                                  "Records", "About"], label_visibility="collapsed")
        st.divider()
        st.markdown(
            f"<div style='font-size:.78rem;color:{INK_SOFT};line-height:1.6'>"
            f"Running on {'GPU' if torch.cuda.is_available() else 'CPU'}<br>"
            f"Random seed {cfg['project']['seed']}</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='dr-disclaimer' style='margin-top:1rem'>"
            "<b>Not a medical device.</b><br>Academic prototype. Nothing here is a "
            "diagnosis or medical advice.</div>", unsafe_allow_html=True)

    disclaimer_banner(cfg)

    {"New screening": page_screen, "Patient timeline": page_history,
     "Records": page_database, "About": page_about}[page](cfg, conn, names)


if __name__ == "__main__":
    main()
