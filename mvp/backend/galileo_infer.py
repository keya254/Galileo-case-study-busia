"""On-demand Galileo inference for a Galileo-format GeoTIFF.

This is a lean copy of the Level 2 path in `case_study/level2_cropharvest.py`, kept inside the
MVP so the container does not need the full evaluation package. It:

1. reads a Galileo export (`Dataset._tif_to_array`),
2. rearranges it into CropHarvest's band layout,
3. converts + normalizes it the same way the saved classifier was trained,
4. runs the frozen Galileo encoder and the saved maize k-NN classifier.

The original conversion code lives in `src/eval/cropharvest/cropharvest_eval.py`
(`CropHarvestEvalBase.cropharvest_array_to_normalized_galileo`).
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

import numpy as np
import rasterio
import torch
from einops import repeat


def _find_repo_root() -> Path:
    """Works both in the repo (mvp/backend) and in the container (/app/backend)."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "src" / "galileo.py").exists():
            return candidate
    return here.parents[1]


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.config import NORMALIZATION_DICT_FILENAME  # noqa: E402
from src.data.dataset import (  # noqa: E402
    LOCATION_BANDS,
    SPACE_BAND_GROUPS_IDX,
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
    TIME_BAND_GROUPS_IDX,
    TIME_BANDS,
    Dataset,
    Normalizer,
    to_cartesian,
)
from src.galileo import Encoder  # noqa: E402
from src.masking import MaskedOutput  # noqa: E402
from src.utils import masked_output_np_to_tensor  # noqa: E402

CH_TIMESTEPS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = REPO_ROOT / "data" / "models" / "nano"
CLASSIFIER_PATH = Path(__file__).resolve().parent.parent / "assets" / "maize_classifier.joblib"


def _load_ch_bands() -> list[str]:
    path = REPO_ROOT / "src" / "eval" / "cropharvest" / "bands.py"
    spec = importlib.util.spec_from_file_location("_mvp_ch_bands", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BANDS


CH_BANDS = _load_ch_bands()
CH_BANDS_TO_SPACE_TIME_BANDS = [CH_BANDS.index(s) for s in SPACE_TIME_BANDS if s in CH_BANDS]
SPACE_TIME_BANDS_TO_CH_BANDS = [i for i, s in enumerate(SPACE_TIME_BANDS) if s in CH_BANDS]
CH_BANDS_TO_SPACE_BANDS = [CH_BANDS.index(s) for s in SPACE_BANDS if s in CH_BANDS]
SPACE_BANDS_TO_CH_BANDS = [i for i, s in enumerate(SPACE_BANDS) if s in CH_BANDS]
CH_BANDS_TO_TIME_BANDS = [CH_BANDS.index(s) for s in TIME_BANDS if s in CH_BANDS]
TIME_BANDS_TO_CH_BANDS = [i for i, s in enumerate(TIME_BANDS) if s in CH_BANDS]
LOCATION_BAND_MAPPING = [i for i, x in enumerate(STATIC_BANDS) if x in LOCATION_BANDS]


def galileo_normalizer() -> Normalizer:
    dicts = Dataset.load_normalization_values(REPO_ROOT / "config" / NORMALIZATION_DICT_FILENAME)
    return Normalizer(std=True, normalizing_dicts=dicts, std_multiplier=2)


@torch.no_grad()
def run_encoder(model: Encoder, mo: MaskedOutput) -> np.ndarray:
    mo = MaskedOutput(*[t.to(DEVICE) for t in mo])
    out = model(
        mo.space_time_x.float(),
        mo.space_x.float(),
        mo.time_x.float(),
        mo.static_x.float(),
        mo.space_time_mask,
        mo.space_mask,
        mo.time_mask,
        mo.static_mask,
        mo.months.long(),
        patch_size=1,
    )
    return model.average_tokens(*out[:-1]).cpu().numpy()


def tif_to_ch_array(tif_path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Galileo GeoTIFF -> [H*W, 12, len(CH_BANDS)] in raw units."""
    dso = Dataset._tif_to_array(tif_path)
    st, sp, tx = dso.space_time_x, dso.space_x, dso.time_x
    h, w, t, _ = st.shape
    if t < CH_TIMESTEPS:
        raise ValueError(f"{tif_path.name} has {t} timesteps; need at least {CH_TIMESTEPS}.")
    st, tx = st[:, :, :CH_TIMESTEPS], tx[:CH_TIMESTEPS]
    arr = np.zeros((h * w, CH_TIMESTEPS, len(CH_BANDS)), dtype=np.float32)
    for j, band in enumerate(CH_BANDS):
        if band in SPACE_TIME_BANDS:
            arr[:, :, j] = st[..., SPACE_TIME_BANDS.index(band)].reshape(h * w, CH_TIMESTEPS)
        elif band in TIME_BANDS:
            arr[:, :, j] = tx[:, TIME_BANDS.index(band)][None, :]
        elif band in SPACE_BANDS:
            arr[:, :, j] = sp[..., SPACE_BANDS.index(band)].reshape(h * w, 1)
    return arr, (h, w)


def _pixel_latlons(tif_path: Path) -> np.ndarray:
    with rasterio.open(tif_path) as src:
        rows, cols = np.meshgrid(np.arange(src.height), np.arange(src.width), indexing="ij")
        xs, ys = rasterio.transform.xy(src.transform, rows.ravel(), cols.ravel(), offset="center")
        xs, ys = np.asarray(xs), np.asarray(ys)
        if src.crs is not None and src.crs.to_epsg() not in (None, 4326):
            from rasterio.warp import transform as warp_transform

            xs, ys = warp_transform(src.crs, "EPSG:4326", xs, ys)
            xs, ys = np.asarray(xs), np.asarray(ys)
    return np.stack([ys, xs], axis=-1)


def _array_to_masked_output(array: np.ndarray, latlons: np.ndarray, normalizer: Normalizer):
    array = array[:, :CH_TIMESTEPS]
    b, t, _ = array.shape

    s_t_x = np.zeros((b, t, len(SPACE_TIME_BANDS)), dtype=np.float32)
    s_t_x[:, :, SPACE_TIME_BANDS_TO_CH_BANDS] = array[:, :, CH_BANDS_TO_SPACE_TIME_BANDS]
    s_t_x = repeat(s_t_x, "b t d -> b h w t d", h=1, w=1)
    s_t_m = np.ones((b, 1, 1, t, len(SPACE_TIME_BANDS)), dtype=np.float32)
    s_t_m[:, :, :, :, SPACE_TIME_BANDS_TO_CH_BANDS] = 0
    s_t_m = s_t_m[:, :, :, :, [g[0] for _, g in SPACE_TIME_BANDS_GROUPS_IDX.items()]]

    sp_x = np.zeros((b, t, len(SPACE_BANDS)), dtype=np.float32)
    sp_x[:, :, SPACE_BANDS_TO_CH_BANDS] = array[:, :, CH_BANDS_TO_SPACE_BANDS]
    sp_x = repeat(sp_x[:, 0], "b d -> b h w d", h=1, w=1)
    sp_m = np.ones((b, 1, 1, len(SPACE_BANDS)), dtype=np.float32)
    sp_m[:, :, :, SPACE_BANDS_TO_CH_BANDS] = 0
    sp_m = sp_m[:, :, :, [g[0] for _, g in SPACE_BAND_GROUPS_IDX.items()]]

    t_x = np.zeros((b, t, len(TIME_BANDS)), dtype=np.float32)
    t_x[:, :, TIME_BANDS_TO_CH_BANDS] = array[:, :, CH_BANDS_TO_TIME_BANDS]
    t_m = np.ones((b, t, len(TIME_BANDS)), dtype=np.float32)
    t_m[:, :, TIME_BANDS_TO_CH_BANDS] = 0
    t_m = t_m[:, :, [g[0] for _, g in TIME_BAND_GROUPS_IDX.items()]]

    st_x = np.zeros((b, len(STATIC_BANDS)), dtype=np.float32)
    st_m = np.ones((b, len(STATIC_BAND_GROUPS_IDX)), dtype=np.float32)
    st_m[:, list(STATIC_BAND_GROUPS_IDX).index("location")] = 0
    st_x[:, LOCATION_BAND_MAPPING] = to_cartesian(latlons[:, 0], latlons[:, 1])

    months = np.fmod(np.arange(0, t), 12)
    months = repeat(months, "t -> b t", b=b)

    return masked_output_np_to_tensor(
        normalizer(s_t_x),
        normalizer(sp_x),
        normalizer(t_x),
        normalizer(st_x),
        s_t_m,
        sp_m,
        t_m,
        st_m,
        months,
    )


class InferenceEngine:
    """Loads the encoder + classifier once and reuses them across jobs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Encoder | None = None
        self._normalizer: Normalizer | None = None
        self._classifier = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            self._normalizer = galileo_normalizer()
            self._model = Encoder.load_from_folder(MODEL_DIR).to(DEVICE)
            self._model.eval()
            import joblib

            self._classifier = joblib.load(CLASSIFIER_PATH)

    def predict_tif(self, tif_path: Path, batch_size: int = 256) -> np.ndarray:
        """Return a [H, W] maize-probability array for a Galileo-format GeoTIFF."""
        self._ensure_loaded()
        assert self._normalizer is not None and self._model is not None

        array, (h, w) = tif_to_ch_array(tif_path)
        latlons = _pixel_latlons(tif_path)

        embeddings = []
        for i in range(0, len(array), batch_size):
            mo = _array_to_masked_output(
                array[i : i + batch_size], latlons[i : i + batch_size], self._normalizer
            )
            embeddings.append(run_encoder(self._model, mo))
        emb = np.concatenate(embeddings, axis=0)
        prob = self._classifier["model"].predict_proba(emb)[:, 1]
        return prob.reshape(h, w).astype(np.float32)


ENGINE = InferenceEngine()