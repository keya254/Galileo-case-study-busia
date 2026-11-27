"""Earth Engine + Cloud Storage helpers for on-demand Sentinel exports.

Flow: submit a Galileo-format export for the requested polygon and dates, poll the EE batch
task, download the GeoTIFF from a staging bucket, then remove the staged object.

Configuration (environment variables):
    EE_PROJECT  Earth Engine / Google Cloud project id (required for on-demand mode)
    EE_BUCKET   GCS bucket used to stage exports (created on first use if missing)

The credentials are the same ones created by `earthengine authenticate` (mounted into the
container in docker-compose). For production, use a dedicated service account and bucket.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import ee
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "src" / "galileo.py").exists():
            return candidate
    return here.parents[1]


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.earthengine.eo import create_ee_image, ee_safe_str  # noqa: E402

_INIT_LOCK = threading.Lock()
_INITIALIZED: set[str] = set()


def _ensure_credentials_file() -> None:
    """Allow cloud deployments to pass the earthengine credentials as a secret env var."""
    raw = os.getenv("EE_CREDENTIALS_JSON", "").strip()
    if not raw:
        return
    path = Path(os.path.expanduser("~/.config/earthengine/credentials"))
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw)
    path.chmod(0o600)


def project() -> str:
    value = os.getenv("EE_PROJECT", "").strip()
    if not value:
        raise RuntimeError(
            "On-demand Sentinel needs EE_PROJECT (your Earth Engine / Google Cloud project id)."
        )
    return value


def bucket_name() -> str:
    value = os.getenv("EE_BUCKET", "").strip()
    if not value:
        raise RuntimeError("On-demand Sentinel needs EE_BUCKET (a GCS bucket for staging exports).")
    return value


def on_demand_configured() -> bool:
    return bool(os.getenv("EE_PROJECT", "").strip() and os.getenv("EE_BUCKET", "").strip())


def init_ee() -> None:
    _ensure_credentials_file()
    proj = project()
    with _INIT_LOCK:
        if proj in _INITIALIZED:
            return
        ee.Initialize(project=proj)
        _INITIALIZED.add(proj)


def _access_token() -> str:
    _ensure_credentials_file()
    from ee.oauth import TOKEN_URI, get_credentials_arguments
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    args = get_credentials_arguments()
    creds = Credentials(
        token=None,
        refresh_token=args["refresh_token"],
        token_uri=TOKEN_URI,
        client_id=args["client_id"],
        client_secret=args["client_secret"],
        scopes=args["scopes"],
    )
    creds.refresh(Request())
    return creds.token


def _request(url: str, method: str = "GET", body: dict | None = None) -> bytes:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Authorization": f"Bearer {_access_token()}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read()


def ensure_bucket() -> str:
    bucket = bucket_name()
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}"
    try:
        _request(url)
        return bucket
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    _request(
        f"https://storage.googleapis.com/storage/v1/b?project={project()}",
        method="POST",
        body={"name": bucket, "location": "US"},
    )
    return bucket


def _export_name(geometry: BaseGeometry, start: date, end: date, prefix: str) -> str:
    min_lon, min_lat, max_lon, max_lat = geometry.bounds
    safe_prefix = "".join(c for c in prefix if c.isalnum())[:16] or "aoi"
    return (
        f"mvp_{safe_prefix}_min_lat={min_lat:.4f}_min_lon={min_lon:.4f}"
        f"_max_lat={max_lat:.4f}_max_lon={max_lon:.4f}"
        f"_dates={start.isoformat()}_{end.isoformat()}"
    )


def submit_export(geometry: dict, start: date, end: date, prefix: str) -> dict:
    """Start an EE export and return {task_id, name, bucket}."""
    init_ee()
    bucket = ensure_bucket()
    geom = shape(geometry)
    ee_polygon = ee.Geometry(geometry)
    name = _export_name(geom, start, end, prefix)
    image = create_ee_image(ee_polygon, start, end)
    task = ee.batch.Export.image.toCloudStorage(
        image=image.clip(ee_polygon),
        description=ee_safe_str(name)[:100],
        bucket=bucket,
        fileNamePrefix=name,
        region=ee_polygon,
        scale=10,
        maxPixels=1e13,
    )
    task.start()
    return {"task_id": task.id, "name": name, "bucket": bucket}


def wait_for_task(task_id: str, timeout: int = 1800, on_status=None) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = ee.data.getTaskStatus(task_id)[0]
        state = status.get("state", "UNKNOWN")
        if on_status:
            on_status(state, status.get("description", ""))
        if state == "COMPLETED":
            return
        if state in ("FAILED", "CANCELLED"):
            raise RuntimeError(status.get("error_message") or f"Earth Engine task {state}")
        time.sleep(5)
    raise TimeoutError(f"Earth Engine export did not finish within {timeout} seconds")


def _list_objects(bucket: str, prefix: str) -> list[dict]:
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?prefix={urllib.parse.quote(prefix)}"
    payload = json.loads(_request(url))
    return payload.get("items", [])


def download_and_cleanup(job: dict, dest: Path) -> Path:
    bucket, name = job["bucket"], job["name"]
    objects = [o for o in _list_objects(bucket, name) if o["name"].endswith(".tif")]
    if not objects:
        raise RuntimeError(f"Earth Engine finished but no GeoTIFF found for {name}")
    # Small AOIs produce one file; if EE sharded the export, take the first shard.
    obj = sorted(objects, key=lambda o: o["name"])[0]
    quoted = urllib.parse.quote(obj["name"], safe="")
    media = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{quoted}?alt=media"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
        urllib.request.Request(media, headers={"Authorization": f"Bearer {_access_token()}"}),
        timeout=300,
    ) as response, open(dest, "wb") as fh:
        fh.write(response.read())
    for item in objects:
        q = urllib.parse.quote(item["name"], safe="")
        try:
            _request(f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{q}", method="DELETE")
        except Exception:  # noqa: BLE001 - cleanup is best-effort
            pass
    return dest