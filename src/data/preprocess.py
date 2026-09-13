"""
Image preprocessing for fundus photographs.
File location: <project_root>/src/data/preprocess.py

Objective
---------
Turn a raw APTOS fundus photograph (anywhere from 640x480 to 4288x2848, a
retinal disc floating on a black rectangle) into a clean, fixed-size
300x300 RGB image suitable for a pretrained CNN.

The pipeline, in order:

    read -> crop black borders -> pad to square -> [optional CLAHE] -> resize

Why this order
--------------
1. **Crop first.** The black border is not information. Worse, the amount of
   black varies per image, so resizing without cropping means the retina
   occupies a different fraction of each 300x300 frame - the network would see
   identical lesions at different scales.
2. **Pad to square before resizing.** A cropped fundus is roughly circular but
   rarely exactly square. Resizing a 1200x1000 crop straight to 300x300
   stretches it horizontally. Padding to 1200x1200 first preserves the aspect
   ratio, so a round optic disc stays round.
3. **CLAHE before resize**, if used, so the contrast enhancement sees the full
   detail rather than an already-downsampled image.
4. **Normalisation is NOT done here.** Cached images stay as uint8 PNGs;
   ImageNet normalisation happens at tensor level in ``transforms.py``. Storing
   float32 normalised arrays would use ~12x more disk for no benefit.

Colour convention
-----------------
OpenCV reads images as **BGR**. Every function here takes and returns BGR
uint8 arrays, and only :func:`load_and_preprocess` has a ``to_rgb`` flag for
the point where you hand the image to PyTorch or matplotlib. Keeping one
convention internally avoids the classic "why are my retinas blue" bug.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. Cropping
# ---------------------------------------------------------------------------
def crop_black_borders(
    image: np.ndarray,
    tol: int = 7,
    min_frac: float = 0.0,
) -> np.ndarray:
    """Remove the black surround from a fundus photograph.

    Builds a mask of "not black" pixels, then crops to that mask's bounding box.

    Parameters
    ----------
    image:
        BGR uint8 image, shape (H, W, 3).
    tol:
        A pixel counts as black if its grayscale value is <= ``tol``.
        7 works well for APTOS; the notebook lets you verify this visually.
        Too low and you keep dark border; too high and you eat into the retina.
    min_frac:
        Fraction of a row (or column) that must be non-black for that row to be
        kept. 0.0 keeps a row if *any* pixel is bright, which is simple but can
        be defeated by a single hot pixel or a bright timestamp in the corner.
        Raising it to ~0.01 makes the crop robust to stray bright pixels at the
        cost of possibly trimming a sliver of retina.

    Returns
    -------
    The cropped image. If the mask is empty (a fully black image), the input is
    returned unchanged rather than raising - one pathological file should not
    stop a 3,662-image batch job.
    """
    if image is None or image.size == 0:
        raise ValueError("crop_black_borders received an empty image.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel image, got shape {image.shape}.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > tol

    if not mask.any():
        # Entirely black - nothing to crop.
        return image

    if min_frac > 0.0:
        rows_keep = mask.mean(axis=1) > min_frac
        cols_keep = mask.mean(axis=0) > min_frac
    else:
        rows_keep = mask.any(axis=1)
        cols_keep = mask.any(axis=0)

    if not rows_keep.any() or not cols_keep.any():
        # min_frac was too aggressive for this image; fall back to a plain crop.
        rows_keep = mask.any(axis=1)
        cols_keep = mask.any(axis=0)

    row_idx = np.where(rows_keep)[0]
    col_idx = np.where(cols_keep)[0]
    return image[row_idx[0] : row_idx[-1] + 1, col_idx[0] : col_idx[-1] + 1]


def black_fraction(image: np.ndarray, tol: int = 7) -> float:
    """Fraction of pixels that are (near) black. Used to quantify wasted area."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float((gray <= tol).mean())


# ---------------------------------------------------------------------------
# 2. Squaring
# ---------------------------------------------------------------------------
def pad_to_square(image: np.ndarray, value: int = 0) -> np.ndarray:
    """Pad the shorter side with ``value`` so the result is square.

    Padding is split evenly, so the retina stays centred. Uses black (0) by
    default, which matches the surround the camera already produces.
    """
    h, w = image.shape[:2]
    if h == w:
        return image

    size = max(h, w)
    pad_v, pad_h = size - h, size - w
    top, bottom = pad_v // 2, pad_v - pad_v // 2
    left, right = pad_h // 2, pad_h - pad_h // 2

    return cv2.copyMakeBorder(
        image, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=(value, value, value),
    )


# ---------------------------------------------------------------------------
# 3. Contrast enhancement (optional)
# ---------------------------------------------------------------------------
def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: int = 8,
) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation on the lightness channel.

    Plain histogram equalisation would shift the colours, which matters here -
    haemorrhages are dark red, exudates are pale yellow, and the model uses
    that. So we convert BGR -> LAB, equalise only **L** (lightness), and convert
    back. Hue and saturation are untouched.

    ``clip_limit`` caps how much local contrast can be amplified; without it,
    CLAHE amplifies sensor noise in the dark periphery into fake "lesions".
    2.0 is conservative. The notebook shows 1.0 / 2.0 / 4.0 side by side.

    Whether CLAHE actually helps is an empirical question, which is why
    ``use_clahe`` is a config flag and an ablation in Milestone 6 - not an
    assumption.
    """
    if clip_limit <= 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_chan, b_chan = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=(int(tile_grid), int(tile_grid)))
    lightness = clahe.apply(lightness)

    return cv2.cvtColor(cv2.merge((lightness, a_chan, b_chan)), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# 4. Resizing
# ---------------------------------------------------------------------------
def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """Resize to ``size`` x ``size``.

    Interpolation is chosen by direction: INTER_AREA for downscaling (it
    averages over the source pixels, so fine detail is anti-aliased rather than
    aliased away) and INTER_LINEAR for the rare upscale. Using INTER_LINEAR to
    shrink a 4288px image to 300px throws away most pixels and produces
    speckle - a subtle bug that quietly costs accuracy.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}.")
    h, w = image.shape[:2]
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interp)


# ---------------------------------------------------------------------------
# 5. The full pipeline
# ---------------------------------------------------------------------------
def preprocess_image(
    image: np.ndarray,
    size: int = 300,
    crop: bool = True,
    tol: int = 7,
    min_frac: float = 0.0,
    square: bool = True,
    clahe: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: int = 8,
) -> np.ndarray:
    """Apply the whole pipeline to one BGR uint8 image. Returns BGR uint8."""
    out = image
    if crop:
        out = crop_black_borders(out, tol=tol, min_frac=min_frac)
    if square:
        out = pad_to_square(out)
    if clahe:
        out = apply_clahe(out, clip_limit=clahe_clip_limit, tile_grid=clahe_tile_grid)
    return resize_image(out, size)


def preprocess_from_config(image: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """Same as :func:`preprocess_image`, reading every setting from the config.

    Use this everywhere (caching, the app, the notebooks) so training and
    inference can never drift apart - the single most common cause of "it
    scored 0.85 in my notebook but the app predicts nonsense".
    """
    pp = cfg["preprocessing"]
    return preprocess_image(
        image,
        size=pp["image_size"],
        crop=pp.get("crop_black_borders", True),
        tol=pp.get("border_tol", 7),
        min_frac=pp.get("border_min_frac", 0.0),
        square=pp.get("pad_to_square", True),
        clahe=pp.get("use_clahe", False),
        clahe_clip_limit=pp.get("clahe_clip_limit", 2.0),
        clahe_tile_grid=pp.get("clahe_tile_grid", 8),
    )


# ---------------------------------------------------------------------------
# 6. I/O helpers
# ---------------------------------------------------------------------------
def read_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR uint8, raising a clear error if it cannot be decoded.

    ``cv2.imread`` returns None rather than raising on a corrupt or missing
    file, and also fails silently on non-ASCII paths on Windows - hence the
    numpy/imdecode fallback.
    """
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None and path.is_file():
        # Windows + non-ASCII path fallback.
        try:
            buf = np.fromfile(str(path), dtype=np.uint8)
            image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except OSError:
            image = None

    if image is None:
        raise ValueError(
            f"Could not read image: {path}\n"
            "The file may be missing, corrupt, or not a real image."
        )
    return image


def load_and_preprocess(
    path: str | Path,
    cfg: Dict[str, Any],
    to_rgb: bool = False,
) -> np.ndarray:
    """Read a file from disk and run the configured pipeline.

    Set ``to_rgb=True`` when the result goes to matplotlib or PyTorch.
    """
    out = preprocess_from_config(read_image(path), cfg)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB) if to_rgb else out


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Small named helper so notebooks never have to remember the flag order."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def cache_dir_for(cfg: Dict[str, Any], root: Path) -> Path:
    """Where cached preprocessed images live, e.g. data/interim/train_300."""
    template = cfg["preprocessing"].get("cache_dir_template", "train_{size}")
    return root / template.format(size=cfg["preprocessing"]["image_size"])


def crop_stats(image: np.ndarray, tol: int = 7) -> Tuple[float, float]:
    """Return (black fraction before crop, black fraction after crop).

    Used by the notebook to put a number on how much area the crop reclaims.
    """
    before = black_fraction(image, tol=tol)
    after = black_fraction(crop_black_borders(image, tol=tol), tol=tol)
    return before, after
