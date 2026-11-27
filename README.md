# Galileo Crop Intelligence - deploy branch

Standalone deployment branch for the Galileo crop MVP. It contains only what the service needs:
the FastAPI backend, the Leaflet frontend, the cached demo assets, and the minimal Galileo
runtime (`src/`, `config/`, `data/models/nano/`). The case study, notebooks, tests and training
data are not included.

## Features

- Search a place, draw a polygon/rectangle, or analyze the map view.
- Cached Busia 3 x 3 km 2019 tile: instant.
- Any other AOI/period: on-demand Sentinel export from Earth Engine, frozen Galileo + saved
  CropHarvest k-NN classifier, cached on disk.
- Maize probability, hectares/acres, confidence, insurance proxy, probability overlay.
- History and change-over-time for the same AOI across periods.

## Run with Docker

```bash
cp .env.example .env    # set EE_PROJECT and EE_BUCKET
docker compose up --build
```

Open http://localhost:8000.

## Deploy to a cloud platform

Build the included `Dockerfile` (port 8000). Set these environment variables:

| Variable | Required | Meaning |
| --- | --- | --- |
| `EE_PROJECT` | for on-demand | Earth Engine / Google Cloud project id |
| `EE_BUCKET` | for on-demand | GCS staging bucket (created on first use) |
| `EE_CREDENTIALS_JSON` | for on-demand | Contents of the `earthengine authenticate` credentials JSON |
| `MVP_MAX_AREA_KM2` | no | On-demand AOI cap (default 9) |
| `MVP_BATCH_SIZE` | no | Inference batch size (default 256) |
| `MVP_DATA_DIR` | no | Cache directory (default `/app/mvp_data`) |

Without `EE_PROJECT`/`EE_BUCKET` the app still serves the cached Busia 2019 demo; on-demand is
reported as off.

`EE_CREDENTIALS_JSON` is written to `~/.config/earthengine/credentials` on startup if no
credentials file is mounted. Copy the contents of your local
`~/.config/earthengine/credentials` (it contains a refresh token) into a platform secret. For a
production service, prefer a dedicated Google service account and bucket.

The analyses cache lives in `mvp_data/`. Mount a persistent volume there so history survives
restarts and redeploys.

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

## Limits

- On-demand AOIs are capped at 9 km2 (CPU inference); a small polygon takes about a minute.
- Periods must span at least 360 days so the classifier gets 12 monthly steps.
- Earth Engine is free for noncommercial use. For commercial use, use the Copernicus pipeline.
- The insurance score is a proxy, not an underwriting decision. Validate with field labels.

## Licence

MIT, same as the parent Galileo repository.
