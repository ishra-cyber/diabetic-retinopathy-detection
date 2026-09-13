"""
Configuration loading utilities.

Every script in this project starts by calling :func:`load_config`. That gives
a single dictionary holding all paths and hyper-parameters, and guarantees that
paths are absolute no matter which directory you run the script from.

Usage
-----
    from src.utils.config import load_config, get_path

    cfg = load_config()                 # reads configs/config.yaml
    raw = get_path(cfg, "raw_dir")      # -> PosixPath('/abs/path/data/raw')
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

# Project root = two levels up from this file (src/utils/config.py -> project/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load the YAML config file into a plain dictionary.

    Parameters
    ----------
    config_path:
        Optional path to a YAML file. Defaults to ``configs/config.yaml``.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist (usually means you are running from
        an unexpected directory or the repo is incomplete).
    ValueError
        If the YAML parses to something other than a mapping.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found at: {path}\n"
            f"Expected the project root to be: {PROJECT_ROOT}\n"
            "Run scripts from the project root, e.g. `python -m scripts.01_inspect_dataset`."
        )

    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} did not parse into a mapping (got {type(cfg)!r}).")

    # Keep the resolved root around so downstream code never has to guess.
    cfg["_project_root"] = str(PROJECT_ROOT)
    cfg["_config_path"] = str(path)
    return cfg


def get_path(cfg: Dict[str, Any], key: str, *parts: str) -> Path:
    """Resolve a key from ``cfg['paths']`` into an absolute Path.

    Extra ``parts`` are appended, so
    ``get_path(cfg, "aptos_dir", "train.csv")`` gives the CSV path.
    """
    paths = cfg.get("paths", {})
    if key not in paths:
        raise KeyError(
            f"'{key}' is not defined under `paths:` in {cfg.get('_config_path')}. "
            f"Available keys: {sorted(paths)}"
        )
    root = Path(cfg["_project_root"])
    return root.joinpath(paths[key], *parts)


def ensure_dirs(cfg: Dict[str, Any], *keys: str) -> None:
    """Create the directories behind the given path keys if they do not exist."""
    for key in keys:
        get_path(cfg, key).mkdir(parents=True, exist_ok=True)


def class_names(cfg: Dict[str, Any]) -> list[str]:
    """Return class names ordered by label index 0..4."""
    mapping = cfg["dataset"]["class_names"]
    return [mapping[i] for i in sorted(mapping)]


# ---------------------------------------------------------------------------
# Experiment overrides (Milestones 4-6)
# ---------------------------------------------------------------------------
def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Nested dicts are merged key by key rather than replaced wholesale, so an
    experiment file can change ``training.batch_size`` without having to repeat
    every other training setting.
    """
    import copy

    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_experiment_config(name: str, config_path: str | os.PathLike | None = None
                           ) -> Dict[str, Any]:
    """Load the base config and merge ``configs/experiments/<name>.yaml` on top.

    The merged dict records ``_experiment`` (the name) so every artefact the run
    writes can be traced back to the file that produced it.
    """
    cfg = load_config(config_path)
    exp_path = Path(cfg["_project_root"]) / "configs" / "experiments" / f"{name}.yaml"

    if not exp_path.is_file():
        available = sorted(p.stem for p in exp_path.parent.glob("*.yaml")) \
            if exp_path.parent.is_dir() else []
        raise FileNotFoundError(
            f"Experiment config not found: {exp_path}\n"
            f"Available experiments: {available or '(none)'}"
        )

    with exp_path.open("r", encoding="utf-8") as fh:
        override = yaml.safe_load(fh) or {}

    merged = deep_update(cfg, override)
    merged["_experiment"] = name
    merged["_experiment_path"] = str(exp_path)
    return merged


def save_config_snapshot(cfg: Dict[str, Any], out_path: str | os.PathLike) -> None:
    """Write the fully-merged config beside a run's outputs.

    This is what makes a result reproducible six weeks later: the checkpoint and
    the exact settings that produced it live in the same folder.
    """
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    clean["_meta"] = {k: v for k, v in cfg.items() if k.startswith("_")}
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(clean, fh, sort_keys=False, default_flow_style=False)
