"""
MILESTONE 1, STEP 1 - Dataset acquisition.
File location: <project_root>/scripts/m1_download_data.py

Objective
---------
Download and extract the APTOS 2019 Blindness Detection dataset from Kaggle
into ``data/raw/aptos2019/`` using your own Kaggle API credentials.

Prerequisites (one time)
------------------------
1. Create a Kaggle account and open
   https://www.kaggle.com/competitions/aptos2019-blindness-detection
2. Click "Join Competition" / accept the competition rules.
   The API returns 403 Forbidden until you do this - it is the single most
   common failure here.
3. Kaggle -> Account -> "Create New API Token" -> downloads kaggle.json
4. Place it at  ~/.kaggle/kaggle.json  and restrict permissions:
       mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
       chmod 600 ~/.kaggle/kaggle.json

Run
---
    python -m scripts.m1_download_data
    python -m scripts.m1_download_data --force       # re-download even if present
    python -m scripts.m1_download_data --keep-zip    # keep the .zip after extraction

The download is roughly 10 GB compressed; allow ~25 GB free disk during
extraction. This script is safe to re-run: it skips work already done.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# --- make `src` importable no matter how this script is launched ---------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import get_path, load_config
from src.utils.logging_utils import get_logger

LOG = get_logger("m1.download")


# ---------------------------------------------------------------------------
# Credential checks
# ---------------------------------------------------------------------------
def check_credentials() -> Path:
    """Verify kaggle.json exists and is readable. Returns its path.

    Kaggle also accepts the KAGGLE_USERNAME / KAGGLE_KEY environment
    variables; if those are set we accept that instead of the file.
    """
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        LOG.info("Using KAGGLE_USERNAME / KAGGLE_KEY from the environment.")
        return Path("<env>")
    if os.environ.get("KAGGLE_API_TOKEN"):
        LOG.info("Using KAGGLE_API_TOKEN from the environment.")
        return Path("<env>")

    kaggle_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    cred = kaggle_dir / "kaggle.json"          # legacy key - what this project expects
    token = kaggle_dir / "access_token"        # modern API token, also supported by the CLI

    if token.is_file() and not cred.is_file():
        LOG.info("Using modern API token: %s", token)
        return token

    if not cred.is_file():
        raise FileNotFoundError(
            f"Kaggle credentials not found at {cred}.\n"
            "Fix (easiest route):\n"
            "  1. Go to https://www.kaggle.com/settings/api\n"
            "  2. Under 'Legacy API Credentials', click 'Create Legacy API Key'\n"
            "     (this downloads kaggle.json; the newer 'API Tokens' option works too\n"
            "      but stores an access_token file instead)\n"
            "  3. Move kaggle.json to ~/.kaggle/  (Windows: %USERPROFILE%\\.kaggle\\)\n"
            "  4. chmod 600 ~/.kaggle/kaggle.json   (not needed on Windows)"
        )

    # Kaggle's client warns loudly if the file is world-readable.
    mode = cred.stat().st_mode & 0o777
    if mode != 0o600:
        LOG.warning("Permissions on %s are %o; running `chmod 600`.", cred, mode)
        try:
            cred.chmod(0o600)
        except OSError as exc:  # e.g. on some mounted filesystems
            LOG.warning("Could not chmod credentials (%s). Continuing anyway.", exc)

    LOG.info("Kaggle credentials found: %s", cred)
    return cred


def check_kaggle_cli() -> None:
    """Make sure the `kaggle` command is importable/installed."""
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "The `kaggle` command was not found on PATH.\n"
            "Install it with:  pip install kaggle\n"
            "If it is installed but not on PATH, ensure your virtualenv is active."
        )


# ---------------------------------------------------------------------------
# Download + extract
# ---------------------------------------------------------------------------
def download_competition(slug: str, dest: Path, force: bool) -> Path:
    """Download the competition archive into ``dest``. Returns the zip path."""
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / f"{slug}.zip"

    if zip_path.is_file() and not force:
        size_gb = zip_path.stat().st_size / 1e9
        LOG.info("Archive already present (%.2f GB): %s - skipping download.", size_gb, zip_path)
        return zip_path

    LOG.info("Downloading competition '%s' into %s (this takes a while)...", slug, dest)
    cmd = ["kaggle", "competitions", "download", "-c", slug, "-p", str(dest)]
    if force:
        cmd.append("--force")

    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to launch the kaggle CLI: {exc}") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        hint = ""
        if "403" in stderr or "Forbidden" in stderr:
            hint = (
                "\nHINT: 403 almost always means you have not accepted the competition "
                "rules. Open the competition page in a browser, click 'Join Competition', "
                "then re-run this script."
            )
        raise RuntimeError(f"kaggle download failed (exit {result.returncode}):\n{stderr}{hint}")

    if not zip_path.is_file():
        # Some CLI versions name the archive differently; find any new zip.
        candidates = sorted(dest.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"Download reported success but no .zip found in {dest}.")
        zip_path = candidates[0]

    LOG.info("Downloaded: %s (%.2f GB)", zip_path, zip_path.stat().st_size / 1e9)
    return zip_path


def extract_archive(zip_path: Path, out_dir: Path, force: bool) -> None:
    """Extract the competition zip into ``out_dir``."""
    marker = out_dir / ".extracted"
    if marker.is_file() and not force:
        LOG.info("Archive already extracted (marker %s exists) - skipping.", marker)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Extracting %s -> %s", zip_path.name, out_dir)

    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.infolist()
            total = len(members)
            for i, member in enumerate(members, start=1):
                zf.extract(member, out_dir)
                if i % 500 == 0 or i == total:
                    LOG.info("  extracted %d/%d files", i, total)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"{zip_path} is not a valid zip (download may have been truncated). "
            "Delete it and re-run with --force."
        ) from exc

    marker.write_text("extracted\n", encoding="utf-8")
    LOG.info("Extraction complete.")


def summarise(aptos_dir: Path) -> None:
    """Print what actually landed on disk, so you can eyeball it immediately."""
    LOG.info("Contents of %s:", aptos_dir)
    for entry in sorted(aptos_dir.iterdir()):
        if entry.is_dir():
            n = sum(1 for _ in entry.iterdir())
            LOG.info("  [dir ] %-20s %d files", entry.name, n)
        else:
            LOG.info("  [file] %-20s %.1f KB", entry.name, entry.stat().st_size / 1024)


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Download APTOS 2019 from Kaggle.")
    parser.add_argument("--force", action="store_true", help="re-download and re-extract")
    parser.add_argument("--keep-zip", action="store_true", help="do not delete the archive")
    args = parser.parse_args()

    cfg = load_config()
    slug = cfg["dataset"]["name"]
    raw_dir = get_path(cfg, "raw_dir")
    aptos_dir = get_path(cfg, "aptos_dir")

    try:
        check_kaggle_cli()
        check_credentials()
        zip_path = download_competition(slug, raw_dir, args.force)
        extract_archive(zip_path, aptos_dir, args.force)
        summarise(aptos_dir)

        if not args.keep_zip:
            LOG.info("Removing archive to reclaim disk (%s). Use --keep-zip to keep it.",
                     zip_path.name)
            zip_path.unlink(missing_ok=True)

    except Exception as exc:  # noqa: BLE001 - top-level CLI handler
        LOG.error("%s", exc)
        return 1

    LOG.info("Done. Next: python -m scripts.m1_inspect_dataset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
