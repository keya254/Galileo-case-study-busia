"""On-disk cache and history for MVP analyses.

Every analysis is keyed by the normalized AOI geometry and the date range, so revisiting the
same polygon/period is instant and the same AOI can be compared across periods.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from shapely.geometry import shape

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ["MVP_DATA_DIR"]) if os.getenv("MVP_DATA_DIR") else BASE_DIR / "mvp_data"
ANALYSES_DIR = DATA_DIR / "analyses"


def aoi_key(geometry: dict) -> str:
    geom = shape(geometry).normalize()
    return hashlib.sha1(geom.wkb_hex.encode()).hexdigest()[:16]


def period_key(start: str, end: str) -> str:
    return f"{start}_{end}"


def result_dir(aoi: str, period: str) -> Path:
    return ANALYSES_DIR / aoi / period


def overlay_path(aoi: str, period: str) -> Path:
    return result_dir(aoi, period) / "overlay.png"


def probability_path(aoi: str, period: str) -> Path:
    return result_dir(aoi, period) / "probability.tif"


def raw_path(aoi: str, period: str) -> Path:
    return result_dir(aoi, period) / "raw_export.tif"


def load_result(aoi: str, period: str) -> dict | None:
    path = result_dir(aoi, period) / "result.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_result(
    aoi: str,
    period: str,
    result: dict,
    overlay_png: bytes,
    probability_tif: Path | None = None,
) -> dict:
    directory = result_dir(aoi, period)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "overlay.png").write_bytes(overlay_png)
    if probability_tif is not None and probability_tif.exists():
        destination = probability_path(aoi, period)
        if probability_tif.resolve() != destination.resolve():
            shutil.copyfile(probability_tif, destination)
    result = {
        **result,
        "aoi_key": aoi,
        "period": period,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overlay": {
            "bounds": result["overlay"]["bounds"],
            "url": f"/api/analyses/{aoi}/{period}/overlay.png",
        },
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2))
    return result


def list_results(aoi: str) -> list[dict]:
    root = ANALYSES_DIR / aoi
    if not root.exists():
        return []
    results = []
    for path in root.glob("*/result.json"):
        try:
            results.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    results.sort(key=lambda item: (item.get("start", ""), item.get("end", "")))
    return results


def delete_result(aoi: str, period: str) -> None:
    directory = result_dir(aoi, period)
    if directory.exists():
        shutil.rmtree(directory)