"""FastAPI backend for the Galileo crop-intelligence MVP.

Fast path: the cached Busia 2019 probability map (`assets/maize_probability.tif`) answers
instantly. Any other area/period is exported from Earth Engine on demand, run through frozen
Galileo + the saved maize classifier, and cached on disk so revisits and change-over-time
comparisons are instant.
"""

from __future__ import annotations

import io
import os
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from pyproj import Geod
from rasterio.mask import mask as raster_mask
from rasterio.transform import array_bounds
from rasterio.warp import transform_geom
from shapely.geometry import box, shape

from . import ee_source, store
from .galileo_infer import ENGINE

BASE_DIR = Path(__file__).resolve().parent.parent
ASSET = BASE_DIR / "assets" / "maize_probability.tif"
FRONTEND = BASE_DIR / "frontend"

CUTOFF = 0.18
STRICT_CUTOFF = 0.5
DEMO_START = "2019-01-01"
DEMO_END = "2019-12-31"
DEMO_PERIOD = store.period_key(DEMO_START, DEMO_END)
MIN_SPAN_DAYS = 360
GEOD = Geod(ellps="WGS84")

_VIRIDIS = np.array(
    [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
    dtype=np.float64,
)

app = FastAPI(title="Galileo Crop Intelligence MVP", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def frontend_cache_headers(request, call_next):
    response = await call_next(request)
    if request.url.path in ("/", "/index.html", "/app.js", "/style.css"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


class AnalyzeRequest(BaseModel):
    geometry: dict = Field(..., description="GeoJSON Polygon/MultiPolygon in EPSG:4326")
    start: str | None = Field(None, description="Start date YYYY-MM-DD (default 2019-01-01)")
    end: str | None = Field(None, description="End date YYYY-MM-DD (default 2019-12-31)")


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, **fields) -> str:
        job_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "message": "Queued",
                "progress": 0,
                "created_at": now,
                "updated_at": now,
                **fields,
            }
        return job_id

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)
                self._jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                )

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None


JOBS = JobStore()


def _pixel_area_m2(src: rasterio.DatasetReader) -> float:
    t = src.transform
    row, col = src.height // 2, src.width // 2
    xs, ys = [], []
    for dc, dr in ((0, 0), (1, 0), (1, 1), (0, 1)):
        x, y = t * (col + dc, row + dr)
        xs.append(x)
        ys.append(y)
    area, _ = GEOD.polygon_area_perimeter(xs, ys)
    return abs(area)


def _load_demo_meta() -> dict:
    with rasterio.open(ASSET) as src:
        b = src.bounds
        return {
            "name": "Busia county, Kenya - 3 x 3 km demo",
            "bounds": [[b.bottom, b.left], [b.top, b.right]],
            "center": [(b.bottom + b.top) / 2, (b.left + b.right) / 2],
            "cutoff": CUTOFF,
            "strict_cutoff": STRICT_CUTOFF,
            "overlay_url": "/api/overlay.png",
            "pixel_size_m": round(_pixel_area_m2(src) ** 0.5, 1),
            "demo_period": {"start": DEMO_START, "end": DEMO_END},
            "coverage_note": "The 2019 Busia tile is cached; other areas/periods use on-demand Sentinel.",
        }


META = _load_demo_meta()
DEMO_BOUNDS = box(META["bounds"][0][1], META["bounds"][0][0], META["bounds"][1][1], META["bounds"][1][0])
MAX_AREA_KM2 = float(os.getenv("MVP_MAX_AREA_KM2", "9"))
BATCH_SIZE = int(os.getenv("MVP_BATCH_SIZE", "256"))


def _colorize(prob: np.ndarray, valid: np.ndarray, alpha: int = 200) -> Image.Image:
    p = np.clip(np.nan_to_num(prob, nan=0.0), 0.0, 1.0)
    idx = p * (len(_VIRIDIS) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(_VIRIDIS) - 1)
    frac = (idx - lo)[..., None]
    rgb = _VIRIDIS[lo] * (1 - frac) + _VIRIDIS[hi] * frac
    rgba = np.zeros((*p.shape, 4), dtype=np.uint8)
    rgba[..., :3] = rgb.astype(np.uint8)
    rgba[..., 3] = np.where(valid, alpha, 0).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _geodesic_area_m2(geom) -> float:
    if geom.is_empty:
        return 0.0
    area, _ = GEOD.geometry_area_perimeter(geom)
    return abs(area)


def _confidence_band(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.55:
        return "medium"
    return "low"


def _insurance_proxy(share_strict: float, confidence: float) -> dict:
    score = round(100 * (0.65 * share_strict + 0.35 * confidence))
    if score >= 65:
        band = "lower risk"
    elif score >= 40:
        band = "medium risk"
    else:
        band = "higher risk"
    return {
        "score": score,
        "band": band,
        "note": "Proxy for cropland consistency and classification confidence, not an underwriting decision.",
    }


def _validate_geometry(geometry: dict):
    try:
        poly = shape(geometry)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid GeoJSON geometry: {exc}") from exc
    if poly.geom_type not in ("Polygon", "MultiPolygon"):
        raise HTTPException(status_code=400, detail="geometry must be a Polygon or MultiPolygon")
    if poly.is_empty or poly.area == 0:
        raise HTTPException(status_code=400, detail="geometry is empty")
    return poly


def _clip_and_score(raster_path: Path, geometry: dict, mode: str) -> tuple[dict, bytes]:
    """Clip a probability raster to the polygon and compute the user-facing metrics."""
    poly = _validate_geometry(geometry)
    with rasterio.open(raster_path) as src:
        if mode == "demo":
            inter = poly.intersection(box(*src.bounds))
            coverage = inter.area / poly.area if poly.area else 0.0
            if inter.is_empty or coverage == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Polygon is outside the cached Busia demo tile.",
                )
        else:
            coverage = 1.0

        geom = transform_geom("EPSG:4326", src.crs, geometry, precision=8)
        try:
            data, transform = raster_mask(src, [geom], crop=True, filled=False, all_touched=False)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Polygon does not overlap the raster") from exc

        arr = data[0]
        mask = np.ma.getmaskarray(arr)
        values = np.asarray(arr.data, dtype=np.float64)
        valid = (~mask) & np.isfinite(values)
        if not valid.any():
            raise HTTPException(status_code=400, detail="No valid model pixels inside that polygon.")

        prob = values[valid]
        n_valid = int(prob.size)
        px_area = _pixel_area_m2(src)
        valid_area_ha = n_valid * px_area / 10_000
        mean_prob = float(prob.mean())
        share_cutoff = float((prob > CUTOFF).mean())
        share_strict = float((prob > STRICT_CUTOFF).mean())
        uncertain = float(((prob >= 0.3) & (prob <= 0.7)).mean())
        confidence = 1.0 - uncertain

        left, bottom, right, top = array_bounds(arr.shape[0], arr.shape[1], transform)
        png = _png_bytes(_colorize(values, valid))

    maize_ha = share_cutoff * valid_area_ha
    core = {
        "area": {
            "polygon_ha": round(_geodesic_area_m2(poly) / 10_000, 2),
            "mapped_ha": round(valid_area_ha, 2),
            "mapped_acres": round(valid_area_ha * 2.47105, 2),
            "maize_ha": round(maize_ha, 2),
            "maize_acres": round(maize_ha * 2.47105, 2),
            "maize_ha_strict": round(share_strict * valid_area_ha, 2),
        },
        "maize": {
            "mean_probability": round(mean_prob, 4),
            "share_above_cutoff": round(share_cutoff, 4),
            "share_above_strict": round(share_strict, 4),
            "cutoff": CUTOFF,
            "strict_cutoff": STRICT_CUTOFF,
        },
        "quality": {
            "coverage_pct": round(100 * coverage, 1),
            "uncertain_share": round(uncertain, 4),
            "confidence": round(confidence, 4),
            "confidence_band": _confidence_band(confidence),
        },
        "insurance_proxy": _insurance_proxy(share_strict, confidence),
        "overlay": {"bounds": [[bottom, left], [top, right]]},
    }
    return core, png


def _build_result(core: dict, geometry: dict, start: str, end: str, source: str) -> dict:
    return {
        "start": start,
        "end": end,
        "source": source,
        "geometry": geometry,
        "area": core["area"],
        "maize": core["maize"],
        "quality": core["quality"],
        "insurance_proxy": core["insurance_proxy"],
        "overlay": {"bounds": core["overlay"]["bounds"]},
        "meta": {
            "scene": META["name"],
            "pixel_size_m": META["pixel_size_m"],
            "model": "Galileo nano (frozen) + CropHarvest k-NN",
            "cutoff": CUTOFF,
        },
    }


def _is_demo(geometry: dict, start: str, end: str) -> bool:
    if start != DEMO_START or end != DEMO_END:
        return False
    poly = shape(geometry)
    return poly.intersection(DEMO_BOUNDS).area / poly.area >= 0.999


def _run_demo(geometry: dict, aoi: str, period: str) -> dict:
    core, png = _clip_and_score(ASSET, geometry, "demo")
    result = _build_result(core, geometry, DEMO_START, DEMO_END, "demo")
    return store.save_result(aoi, period, result, png)


def _run_job(job_id: str, geometry: dict, start: str, end: str, aoi: str, period: str) -> None:
    directory = store.result_dir(aoi, period)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        JOBS.update(
            job_id,
            status="exporting",
            message="Submitting Sentinel export to Earth Engine...",
            progress=10,
        )
        job = ee_source.submit_export(geometry, date.fromisoformat(start), date.fromisoformat(end), aoi)
        JOBS.update(
            job_id,
            message="Waiting for Earth Engine to build the Sentinel stack (usually 1-3 min)...",
            progress=25,
            task_id=job["task_id"],
        )
        ee_source.wait_for_task(
            job["task_id"],
            on_status=lambda state, desc: JOBS.update(
                job_id, message=f"Earth Engine export: {state.lower()}..."
            ),
        )
        JOBS.update(
            job_id, status="downloading", message="Downloading the Sentinel export...", progress=55
        )
        # Keep the dates=... part of the name: Galileo reads the start month from the filename.
        raw = directory / f"{job['name']}.tif"
        ee_source.download_and_cleanup(job, raw)
        JOBS.update(
            job_id,
            status="embedding",
            message="Running frozen Galileo embeddings and the maize classifier...",
            progress=65,
        )
        prob = ENGINE.predict_tif(raw, batch_size=BATCH_SIZE)

        prob_tif = store.probability_path(aoi, period)
        with rasterio.open(raw) as src:
            profile = src.profile.copy()
        profile.update(count=1, dtype="float32", nodata=float("nan"), compress="deflate")
        with rasterio.open(prob_tif, "w", **profile) as dst:
            dst.write(prob.astype("float32"), 1)

        JOBS.update(
            job_id, status="scoring", message="Computing maize area and confidence...", progress=90
        )
        core, png = _clip_and_score(prob_tif, geometry, "ondemand")
        result = _build_result(core, geometry, start, end, "sentinel")
        saved = store.save_result(aoi, period, result, png, probability_tif=prob_tif)
        if os.getenv("MVP_KEEP_RAW", "0") != "1":
            raw.unlink(missing_ok=True)
        JOBS.update(job_id, status="done", message="Done", progress=100, result=saved)
    except Exception as exc:  # noqa: BLE001
        JOBS.update(job_id, status="error", message=str(exc), error=str(exc), progress=100)


def _full_overlay_png() -> bytes:
    with rasterio.open(ASSET) as src:
        prob = src.read(1)
    return _png_bytes(_colorize(prob, np.isfinite(prob), alpha=170))


_FULL_OVERLAY: bytes | None = None


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "scene": META["name"],
        "on_demand": ee_source.on_demand_configured(),
    }


@app.get("/api/meta")
def meta() -> dict:
    return {
        **META,
        "on_demand": ee_source.on_demand_configured(),
        "max_area_km2": MAX_AREA_KM2,
        "min_span_days": MIN_SPAN_DAYS,
    }


@app.get("/api/overlay.png")
def overlay_png() -> Response:
    global _FULL_OVERLAY
    if _FULL_OVERLAY is None:
        _FULL_OVERLAY = _full_overlay_png()
    return Response(
        content=_FULL_OVERLAY,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    start = req.start or DEMO_START
    end = req.end or DEMO_END
    try:
        start_date, end_date = date.fromisoformat(start), date.fromisoformat(end)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Dates must be YYYY-MM-DD.") from exc
    if end_date <= start_date:
        raise HTTPException(status_code=400, detail="end must be after start.")
    if (end_date - start_date).days < MIN_SPAN_DAYS:
        raise HTTPException(
            status_code=400,
            detail=f"Choose at least {MIN_SPAN_DAYS} days (a full crop year) so 12 monthly steps exist.",
        )

    poly = _validate_geometry(req.geometry)
    area_km2 = _geodesic_area_m2(poly) / 1e6
    if not _is_demo(req.geometry, start, end) and area_km2 > MAX_AREA_KM2:
        raise HTTPException(
            status_code=400,
            detail=f"On-demand AOIs are limited to {MAX_AREA_KM2:g} km2 in this MVP; draw a smaller polygon.",
        )

    aoi = store.aoi_key(req.geometry)
    period = store.period_key(start, end)
    cached = store.load_result(aoi, period)
    if cached:
        return {"job_id": None, "status": "done", "result": cached, "cached": True}

    if _is_demo(req.geometry, start, end):
        result = _run_demo(req.geometry, aoi, period)
        return {"job_id": None, "status": "done", "result": result, "cached": False}

    if not ee_source.on_demand_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "On-demand Sentinel is not configured. Set EE_PROJECT and EE_BUCKET, or use the "
                "cached Busia 2019 demo tile."
            ),
        )

    job_id = JOBS.create(aoi_key=aoi, period=period, start=start, end=end)
    thread = threading.Thread(
        target=_run_job, args=(job_id, req.geometry, start, end, aoi, period), daemon=True
    )
    thread.start()
    return {
        "job_id": job_id,
        "status": "queued",
        "aoi_key": aoi,
        "period": period,
        "start": start,
        "end": end,
    }


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found (it may have restarted).")
    return job


@app.get("/api/analyses")
def analyses(aoi_key: str) -> list[dict]:
    return store.list_results(aoi_key)


@app.get("/api/analyses/{aoi}/{period}")
def analysis_detail(aoi: str, period: str) -> dict:
    result = store.load_result(aoi, period)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return result


@app.get("/api/analyses/{aoi}/{period}/overlay.png")
def analysis_overlay(aoi: str, period: str) -> FileResponse:
    path = store.overlay_path(aoi, period)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Overlay not found")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
def api_not_found(path: str) -> dict:
    """Always answer unmatched /api paths with JSON, never the frontend HTML."""
    raise HTTPException(status_code=404, detail="API route not found")


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="static")