from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse
import pandas as pd

import requests


def _safe_filename(name: str) -> str:
    # Keep it filesystem-friendly but readable
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._-]", "", name)
    return name or "unknown"


def _infer_ext(resp: requests.Response, url: str) -> str:
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "image/" in ct:
        ext = ct.split("image/", 1)[1].split(";", 1)[0].strip()
        # normalise a couple of common cases
        if ext == "jpeg":
            return ".jpg"
        if ext in {"png", "jpg", "webp", "gif", "svg+xml"}:
            return ".svg" if ext == "svg+xml" else f".{ext}"
    # fallback: try URL path suffix
    path = urlparse(url).path
    if "." in path:
        suffix = Path(path).suffix
        if suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}:
            return ".jpg" if suffix.lower() == ".jpeg" else suffix.lower()
    return ".png"  # reasonable default for flags


def download_flag_images_from_csv(
    csv_path: str | Path,
    out_dir: str | Path = "flags",
    *,
    country_col: str = "country",
    url_col: str = "flag_source",
    filename_mode: str = "country",  # "country" or "code" (from URL tail)
    timeout_s: float = 20.0,
    delay_s: float = 0.1,
    retries: int = 3,
    overwrite: bool = False,
    user_agent: str = "flag-downloader/1.0",
) -> None:
    """
    Read rows from a CSV with columns: country, flag_source (URL), download images to out_dir.

    filename_mode:
      - "country":  Afghanistan.png
      - "code":     AFG.png   (for FIFA URLs ending in /AFG)
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    mapper = {
        row.original_name: row.replacement_name
        for row in pd.read_csv("fifa_member_to_canonical_name_map.csv").itertuples()
    }

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")
        if country_col not in reader.fieldnames or url_col not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain columns '{country_col}' and '{url_col}'. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            country = (row.get(country_col) or "").strip()
            url = (row.get(url_col) or "").strip()
            if not country or not url:
                continue

            code = url.rstrip("/").split("/")[-1] if url else "UNK"
            base = _safe_filename(mapper.get(country, country)) if filename_mode == "country" else _safe_filename(code)

            # Download (with retries) and only then finalise the extension
            last_err: Optional[Exception] = None
            resp: Optional[requests.Response] = None
            for attempt in range(1, retries + 1):
                try:
                    resp = session.get(url, timeout=timeout_s)
                    resp.raise_for_status()
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if attempt < retries:
                        time.sleep(0.5 * attempt)
            if last_err or resp is None:
                print(f"[FAIL] {country}: {url} ({last_err})")
                continue

            ext = _infer_ext(resp, url)
            out_path = out_dir / f"{base}{ext}"
            if out_path.exists() and not overwrite:
                print(f"[SKIP] {out_path.name} already exists")
                time.sleep(delay_s)
                continue

            # Write atomically
            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            try:
                with tmp_path.open("wb") as wf:
                    wf.write(resp.content)
                tmp_path.replace(out_path)
                print(f"[OK]   {country} -> {out_path}")
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

            time.sleep(delay_s)


# Example usage:
download_flag_images_from_csv("fifa_members_with_flag_sources.csv", out_dir="flags", filename_mode="country")
# download_flag_images_from_csv("flags.csv", out_dir="flags", filename_mode="country", overwrite=False)
