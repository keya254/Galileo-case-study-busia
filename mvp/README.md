# Galileo Crop Intelligence MVP

Search a place, draw a polygon (or analyze the map view), choose a period, and get maize
probability, maize acreage, confidence and an insurance proxy. Run the same area for another
year and the app shows what changed.

Two inference paths:

- **Cached demo** - the Busia 3 x 3 km, 2019 probability map is served instantly.
- **On-demand Sentinel** - any other AOI/period is exported from Earth Engine, run through the
  frozen Galileo encoder + the saved CropHarvest k-NN classifier, then cached on disk.

Every analysis is keyed by the normalized polygon and date range, so revisiting is instant and
the same AOI can be compared across periods.

## Run with Docker (recommended)

Prerequisites: Docker, an Earth Engine account (`earthengine authenticate`), and an EE/GCP project.

```bash
cd mvp
cp .env.example .env      # set EE_PROJECT and a unique EE_BUCKET
docker compose up --build
```

Open http://localhost:8000.

The compose file mounts your Earth Engine credentials read-only and persists analyses in
`mvp/mvp_data`. Change the mount if your credentials live elsewhere.

## Run locally

The repo's `.venv` already has PyTorch (from `uv sync`).

```bash
# from the repo root
.venv/bin/pip install -r mvp/requirements.txt
cd mvp
EE_PROJECT=your-project-id \
EE_BUCKET=galileo-mvp-yourname \
../.venv/bin/uvicorn backend.main:app --port 8000
```

Open http://localhost:8000. Without `EE_PROJECT`/`EE_BUCKET`, the cached Busia 2019 demo still
works and the UI says on-demand is off.

## API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/meta` | Demo bounds, cutoffs, on-demand status, limits |
| GET | `/api/health` | Health check |
| GET | `/api/overlay.png` | Full cached-tile probability overlay |
| POST | `/api/analyze` | `{geometry, start?, end?}` -> done result or queued job |
| GET | `/api/jobs/{job_id}` | Job status/message/progress/result |
| GET | `/api/analyses?aoi_key=...` | History for one AOI |
| GET | `/api/analyses/{aoi}/{period}` | One saved analysis |
| GET | `/api/analyses/{aoi}/{period}/overlay.png` | Saved polygon overlay |

Example (cached 2019 Busia area):

```bash
curl -s http://localhost:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"geometry":{"type":"Polygon","coordinates":[[[34.228,0.472],[34.232,0.472],[34.232,0.476],[34.228,0.476],[34.228,0.472]]]}}'
```

Example (on-demand 2020):

```bash
curl -s http://localhost:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"geometry":{"type":"Polygon","coordinates":[[[34.228,0.472],[34.232,0.472],[34.232,0.476],[34.228,0.476],[34.228,0.472]]]},"start":"2020-01-01","end":"2020-12-31"}'
```

## Limits and notes

- On-demand AOIs are capped at **9 km2** (`MVP_MAX_AREA_KM2`) because Galileo runs on CPU.
  A small polygon finishes in about a minute; a 3 x 3 km tile takes several minutes.
- Periods must span at least **360 days** so the classifier gets 12 monthly steps.
- Earth Engine is free for noncommercial use. For a commercial deployment use the repo's
  Copernicus pipeline instead of EE, and a dedicated service account and bucket.
- The insurance score is a proxy for cropland consistency and confidence, not an underwriting
  model. Validate with field labels before using it in decisions.

## Next steps

1. Move inference to a GPU worker/queue for large AOIs.
2. Add tiling so a 10 x 10 km request runs several small tiles in parallel.
3. Add auth, usage limits, and per-user history.
4. Validate against local field labels; retrain the classifier with a local active-learning loop.

## Licence

Same MIT licence as the parent Galileo repository. Model outputs are indicative only.