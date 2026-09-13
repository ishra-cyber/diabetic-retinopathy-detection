"""
MILESTONE 7 - Grad-CAM explainability figures.
File location: <project_root>/scripts/m7_gradcam.py

Objective
---------
Produce heatmaps showing which regions drove each prediction, for:
  * correctly classified examples of every severity stage
  * FAILURE cases - the model's worst mistakes

Include the failures in your report. A page of only successes reads as
cherry-picking; showing where the model fails, and being honest about it, is
what makes an explainability section credible.

Run
---
    python -m scripts.m7_gradcam --run effnetb3_weighted
    python -m scripts.m7_gradcam                      # picks the best run by test QWK
    python -m scripts.m7_gradcam --per-class 3 --failures 8

Produces
--------
    reports/figures/m7_gradcam_correct.png
    reports/figures/m7_gradcam_failures.png
    reports/m7_gradcam_stats.csv
    app/assets/gradcam/<id>.png            individual overlays, reused by the app
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.preprocess import bgr_to_rgb, cache_dir_for, preprocess_from_config, read_image
from src.data.split import load_split
from src.data.transforms import build_transforms
from src.explain.gradcam import GradCAM, heatmap_statistics, overlay_heatmap
from src.models.evaluate import find_runs, load_checkpoint
from src.utils.config import class_names, ensure_dirs, get_path, load_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

LOG = get_logger("m7")


# ---------------------------------------------------------------------------
def load_preprocessed_rgb(image_id: str, cfg: Dict[str, Any]) -> np.ndarray:
    """Return the 300x300 RGB image for an id, from cache if available."""
    ext = cfg["dataset"]["image_ext"]
    cached = cache_dir_for(cfg, get_path(cfg, "interim_dir")) / f"{image_id}{ext}"
    if cached.is_file():
        return bgr_to_rgb(read_image(cached))
    raw = get_path(cfg, "aptos_dir") / cfg["dataset"]["train_images"] / f"{image_id}{ext}"
    return bgr_to_rgb(preprocess_from_config(read_image(raw), cfg))


def pick_examples(
    predictions_path: Path,
    per_class: int,
    n_failures: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Choose correctly-classified examples per class, and the worst failures.

    "Worst" means largest |predicted - true|, because grading is ordinal: a
    Severe case called No DR is a clinically different failure from one called
    Moderate.
    """
    data = np.load(predictions_path, allow_pickle=True)
    y_true, y_pred = data["y_true"], data["y_pred"]
    probabilities, ids = data["probabilities"], data["ids"]

    confidence = probabilities[np.arange(len(y_pred)), y_pred]
    correct = y_true == y_pred
    distance = np.abs(y_pred.astype(int) - y_true.astype(int))

    def record(i: int) -> Dict[str, Any]:
        return {"id": str(ids[i]), "true": int(y_true[i]), "pred": int(y_pred[i]),
                "confidence": float(confidence[i]), "distance": int(distance[i])}

    corrects: List[Dict[str, Any]] = []
    for label in sorted(set(y_true.tolist())):
        idx = np.where(correct & (y_true == label))[0]
        # Most confident correct predictions - the clearest examples to show.
        idx = idx[np.argsort(-confidence[idx])][:per_class]
        corrects.extend(record(i) for i in idx)

    wrong_idx = np.where(~correct)[0]
    # Sort by error distance, then by confidence: a confidently wrong prediction
    # is the most interesting failure to explain.
    order = np.lexsort((-confidence[wrong_idx], -distance[wrong_idx]))
    failures = [record(i) for i in wrong_idx[order][:n_failures]]

    return corrects, failures


def make_grid(
    model,
    examples: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    device: torch.device,
    names: List[str],
    out_path: Path,
    title: str,
    save_individual_to: Path | None = None,
) -> pd.DataFrame:
    """Render a grid of (original, heatmap, overlay) triplets. Returns stats."""
    transform = build_transforms(cfg, train=False)     # never augment at inference
    alpha = float(cfg.get("explain", {}).get("overlay_alpha", 0.4))

    rows: List[Dict[str, Any]] = []
    n = len(examples)
    if n == 0:
        LOG.warning("No examples to plot for '%s'.", title)
        return pd.DataFrame()

    fig, axes = plt.subplots(n, 3, figsize=(8.4, 2.8 * n))
    axes = np.array(axes).reshape(n, 3)

    cam = GradCAM(model)
    try:
        for r, ex in enumerate(examples):
            image_rgb = load_preprocessed_rgb(ex["id"], cfg)
            tensor = transform(image_rgb).unsqueeze(0).to(device)

            heatmap, predicted, probabilities = cam(tensor)
            overlay = overlay_heatmap(image_rgb, heatmap, alpha=alpha)
            stats = heatmap_statistics(heatmap)

            rows.append({**ex, "cam_predicted": predicted,
                         "cam_confidence": round(float(probabilities[predicted]), 4),
                         **stats})

            axes[r, 0].imshow(image_rgb)
            axes[r, 0].set_title(f"{ex['id']}\ntrue: {ex['true']} {names[ex['true']]}",
                                 fontsize=7)
            axes[r, 1].imshow(heatmap, cmap="jet")
            axes[r, 1].set_title("Grad-CAM", fontsize=7)
            mark = "OK" if ex["true"] == ex["pred"] else f"WRONG (off by {ex['distance']})"
            axes[r, 2].imshow(overlay)
            axes[r, 2].set_title(
                f"pred: {ex['pred']} {names[ex['pred']]}\n"
                f"conf {ex['confidence']:.2f} - {mark}", fontsize=7)
            for c in range(3):
                axes[r, c].axis("off")

            if save_individual_to is not None:
                save_individual_to.mkdir(parents=True, exist_ok=True)
                import cv2
                cv2.imwrite(str(save_individual_to / f"{ex['id']}_cam.png"),
                            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    finally:
        cam.remove_hooks()

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved figure: %s", out_path)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM figures.")
    parser.add_argument("--run", default=None,
                        help="run folder name (default: highest test QWK)")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--per-class", type=int, default=2,
                        help="correctly-classified examples per class")
    parser.add_argument("--failures", type=int, default=6,
                        help="worst misclassifications to show")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    ensure_dirs(cfg, "reports_dir", "figures_dir")
    figures_dir = get_path(cfg, "figures_dir")
    reports_dir = get_path(cfg, "reports_dir")
    gradcam_dir = get_path(cfg, "gradcam_dir")
    names = class_names(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("Device: %s", device)

    # -- choose the run ----------------------------------------------------
    runs = find_runs(get_path(cfg, "experiments_dir"))
    if not runs:
        LOG.error("No trained runs found.")
        return 1

    if args.run:
        matches = [r for r in runs if r.name == args.run]
        if not matches:
            LOG.error("Run '%s' not found. Available: %s",
                      args.run, [r.name for r in runs])
            return 1
        run_dir = matches[0]
    else:
        scored = []
        for r in runs:
            mpath = r / f"metrics_{args.split}.json"
            if mpath.is_file():
                scored.append((json.loads(mpath.read_text()).get("qwk", -1), r))
        if not scored:
            LOG.error("No metrics_%s.json found. Run Milestone 6 first:\n"
                      "   python -m scripts.m6_evaluate", args.split)
            return 1
        scored.sort(reverse=True, key=lambda t: t[0])
        run_dir = scored[0][1]
        LOG.info("Auto-selected best run: %s (%s QWK %.4f)",
                 run_dir.name, args.split, scored[0][0])

    predictions_path = run_dir / f"{args.split}_predictions.npz"
    if not predictions_path.is_file():
        LOG.error("Missing %s. Run Milestone 6 first:\n"
                  "   python -m scripts.m6_evaluate", predictions_path.name)
        return 1

    # -- model -------------------------------------------------------------
    try:
        model, ckpt = load_checkpoint(run_dir / "best.pt", device,
                                      num_classes=cfg["dataset"]["num_classes"])
    except Exception as exc:  # noqa: BLE001
        LOG.error("%s: %s", type(exc).__name__, exc)
        return 1
    LOG.info("Model: %s (checkpoint epoch %s)", ckpt.get("backbone"), ckpt.get("epoch"))

    corrects, failures = pick_examples(predictions_path, args.per_class, args.failures)
    LOG.info("Selected %d correct example(s) and %d failure(s).",
             len(corrects), len(failures))

    stats_correct = make_grid(
        model, corrects, cfg, device, names,
        figures_dir / "m7_gradcam_correct.png",
        f"Grad-CAM: correctly classified examples ({ckpt.get('backbone')}, {args.split} split)",
        save_individual_to=gradcam_dir,
    )
    stats_failures = make_grid(
        model, failures, cfg, device, names,
        figures_dir / "m7_gradcam_failures.png",
        f"Grad-CAM: worst misclassifications ({ckpt.get('backbone')}, {args.split} split)",
        save_individual_to=gradcam_dir,
    )

    stats = pd.concat(
        [stats_correct.assign(group="correct"), stats_failures.assign(group="failure")],
        ignore_index=True,
    )
    stats.to_csv(reports_dir / "m7_gradcam_stats.csv", index=False)
    LOG.info("Saved table: %s", reports_dir / "m7_gradcam_stats.csv")

    # -- the honest check --------------------------------------------------
    if len(stats):
        mean_border = float(stats["border_fraction"].mean())
        LOG.info("-" * 64)
        LOG.info("Mean fraction of high-attention area in the image BORDER: %.1f%%",
                 mean_border * 100)
        if mean_border > 0.35:
            LOG.warning("That is high. The model may be keying on the frame edge rather "
                        "than on retinal pathology. Report this honestly - it is a real "
                        "finding, and a known failure mode of fundus classifiers.")
        else:
            LOG.info("Low border attention - consistent with the model attending to the "
                     "retina rather than the frame.")
        LOG.info("-" * 64)

    LOG.info("Now LOOK at the two figures. The question to answer in your report is "
             "not 'is the heatmap pretty' but 'is it on plausible pathology'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
