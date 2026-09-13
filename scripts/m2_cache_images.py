"""
MILESTONE 2, STEP 2 - Precompute preprocessed images to disk.
File location: <project_root>/scripts/m2_cache_images.py

Objective
---------
Run the (deterministic, expensive) preprocessing pipeline once over all 3,662
training images and save the 300x300 results to ``data/interim/train_300/``.

Why bother
----------
The raw APTOS images go up to 4288x2848. Decoding one of those PNGs and
cropping it costs far more CPU than the GPU needs to do a forward+backward
pass on it. Without caching, your RTX 3050 sits idle waiting for the data
loader, and this happens on EVERY epoch of EVERY run - three architectures
times ~25 epochs means you would pay that cost about 75 times.

Cached 300x300 PNGs decode in a millisecond or two, so the GPU stays fed.
Expect a large speed-up per epoch; measure it yourself with the timing that
Milestone 4 prints.

Cost: roughly 100-200 MB of disk for the cache (vs ~9 GB of raw images).

IMPORTANT
---------
The cache is keyed by image size only, not by the other preprocessing options.
If you change ``crop_black_borders``, ``pad_to_square`` or ``use_clahe`` in the
config, the cache is STALE - rerun with ``--force``. The script prints the
settings it used into ``<cache_dir>/_cache_meta.json`` so you can check.

Run
---
    python -m scripts.m2_cache_images                 # cache everything
    python -m scripts.m2_cache_images --limit 20      # quick trial first
    python -m scripts.m2_cache_images --force         # rebuild from scratch
    python -m scripts.m2_cache_images --workers 4     # parallelism

On Windows, start with ``--limit 20`` to confirm it works, then run the full job.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --- make `src` importable no matter how this script is launched ------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.data.inspect import load_labels
from src.data.preprocess import cache_dir_for, preprocess_from_config, read_image
from src.utils.config import ensure_dirs, get_path, load_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

LOG = get_logger("m2.cache")


# ---------------------------------------------------------------------------
# Worker. Must be a module-level function so it can be pickled and sent to a
# child process - Windows uses 'spawn', not 'fork', so closures do not work.
# ---------------------------------------------------------------------------
def _process_one(task: Tuple[str, str, Dict[str, Any]]) -> Tuple[str, bool, str]:
    """Preprocess one image. Returns (image_id, ok, message)."""
    src, dst, cfg = task
    try:
        image = read_image(src)
        out = preprocess_from_config(image, cfg)
        # PNG compression level 3: near-instant writes, still ~2x smaller than
        # level 0. Level 9 would halve the files again but is ~10x slower.
        ok = cv2.imwrite(dst, out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            return Path(src).stem, False, "cv2.imwrite returned False"
        return Path(src).stem, True, ""
    except Exception as exc:  # noqa: BLE001 - one bad file must not kill the job
        return Path(src).stem, False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
def build_tasks(
    image_ids: List[str],
    images_dir: Path,
    cache_dir: Path,
    ext: str,
    cfg: Dict[str, Any],
    force: bool,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Build the work list, skipping images already cached unless ``force``."""
    tasks = []
    skipped = 0
    for image_id in image_ids:
        dst = cache_dir / f"{image_id}{ext}"
        if dst.is_file() and not force:
            skipped += 1
            continue
        tasks.append((str(images_dir / f"{image_id}{ext}"), str(dst), cfg))
    if skipped:
        LOG.info("Skipping %d image(s) already cached (use --force to rebuild).", skipped)
    return tasks


def write_meta(cache_dir: Path, cfg: Dict[str, Any], n_cached: int) -> None:
    """Record which preprocessing settings produced this cache."""
    meta = {
        "n_images": n_cached,
        "preprocessing": cfg["preprocessing"],
        "seed": cfg["project"]["seed"],
    }
    (cache_dir / "_cache_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def directory_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.glob("*.png")) / 1e6


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Cache preprocessed APTOS images.")
    parser.add_argument("--workers", type=int, default=0,
                        help="parallel processes (0 = auto: min(4, cpu_count))")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N images (for a trial run)")
    parser.add_argument("--force", action="store_true",
                        help="re-process images that are already cached")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["project"]["seed"])

    ds = cfg["dataset"]
    aptos_dir = get_path(cfg, "aptos_dir")
    images_dir = aptos_dir / ds["train_images"]
    train_csv = aptos_dir / ds["train_csv"]

    ensure_dirs(cfg, "interim_dir")
    cache_dir = cache_dir_for(cfg, get_path(cfg, "interim_dir"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    size = cfg["preprocessing"]["image_size"]
    LOG.info("Cache directory : %s", cache_dir)
    LOG.info("Target size     : %dx%d", size, size)
    LOG.info("Crop borders    : %s (tol=%s)",
             cfg["preprocessing"]["crop_black_borders"], cfg["preprocessing"]["border_tol"])
    LOG.info("Pad to square   : %s", cfg["preprocessing"].get("pad_to_square", True))
    LOG.info("CLAHE           : %s", cfg["preprocessing"]["use_clahe"])

    try:
        df = load_labels(train_csv, ds["id_col"], ds["label_col"])
    except Exception as exc:  # noqa: BLE001
        LOG.error("%s", exc)
        return 1

    image_ids = df[ds["id_col"]].tolist()
    if args.limit:
        image_ids = image_ids[: args.limit]
        LOG.info("Trial run: limiting to %d image(s).", len(image_ids))

    tasks = build_tasks(image_ids, images_dir, cache_dir, ds["image_ext"], cfg, args.force)
    if not tasks:
        LOG.info("Nothing to do - cache is already complete (%d files, %.1f MB).",
                 len(list(cache_dir.glob("*.png"))), directory_size_mb(cache_dir))
        return 0

    workers = args.workers if args.workers > 0 else min(4, os.cpu_count() or 1)
    LOG.info("Processing %d image(s) with %d worker process(es)...", len(tasks), workers)

    failures: List[Tuple[str, str]] = []
    done = 0
    start = time.perf_counter()

    if workers <= 1:
        # Sequential: easier to debug, and avoids process overhead for small jobs.
        for task in tasks:
            image_id, ok, msg = _process_one(task)
            done += 1
            if not ok:
                failures.append((image_id, msg))
            if done % 200 == 0 or done == len(tasks):
                rate = done / max(time.perf_counter() - start, 1e-9)
                LOG.info("  %d/%d  (%.1f img/s)", done, len(tasks), rate)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_one, task) for task in tasks]
            for future in as_completed(futures):
                image_id, ok, msg = future.result()
                done += 1
                if not ok:
                    failures.append((image_id, msg))
                if done % 200 == 0 or done == len(tasks):
                    rate = done / max(time.perf_counter() - start, 1e-9)
                    LOG.info("  %d/%d  (%.1f img/s)", done, len(tasks), rate)

    elapsed = time.perf_counter() - start
    n_cached = len(list(cache_dir.glob(f"*{ds['image_ext']}")))
    write_meta(cache_dir, cfg, n_cached)

    LOG.info("-" * 60)
    LOG.info("Processed %d image(s) in %.1f s (%.1f img/s)",
             len(tasks), elapsed, len(tasks) / max(elapsed, 1e-9))
    LOG.info("Cache now holds %d file(s), %.1f MB", n_cached, directory_size_mb(cache_dir))

    if failures:
        LOG.error("%d image(s) FAILED:", len(failures))
        for image_id, msg in failures[:10]:
            LOG.error("  %s: %s", image_id, msg)
        if len(failures) > 10:
            LOG.error("  ... and %d more", len(failures) - 10)
        return 1

    LOG.info("All images cached successfully.")
    LOG.info("Next: Milestone 3 - stratified train/val/test split.")
    return 0


if __name__ == "__main__":
    # The __main__ guard is REQUIRED on Windows: child processes re-import this
    # module, and without the guard they would each re-run main() recursively.
    sys.exit(main())
