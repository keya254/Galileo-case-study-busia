# Galileo case study: Busia county, Kenya

Training materials for working with a **geospatial foundation model** on a real use case.
Participants use **Galileo** on a 3 × 3 km area of Busia county, western Kenya, and work through the
three levels of using a foundation model, from exploring the landscape to mapping maize:

1. **Level 1 — explore the landscape (no labels):** use the embeddings as they are to group the area
   into zones of similar land and find places that look like a chosen location. No labels, no training.
2. **Level 2 — map maize with a small model on top:** keep Galileo fixed and train a simple classifier
   on its embeddings using CropHarvest maize labels.
3. **Level 3 — map maize by fine-tuning:** retrain Galileo itself for the maize task and compare the
   result with Level 2.

## Start here

**[case_study/README.md](case_study/README.md)**: setup, data export and the
step-by-step instructions for all three levels.

Quick setup on Windows (in **Miniforge Prompt**):

```
git clone https://github.com/jc-trainings/galileo.git
cd galileo
mamba env create -f environment.yml
conda activate galileo-gfm
```

## What is in this repository

| Path                                           | Contents                                                                                                        |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `case_study/`                                  | **The case study**: data export, landscape exploration (Level 1), maize mapping (Levels 2–3), reference results |
| `environment.yml`                              | Miniforge/conda environment (Windows, macOS, Linux)                                                             |
| `data/models/nano/`                            | Pretrained Galileo nano weights (about 1 million parameters; runs on a laptop CPU)                              |
| `data/busia/`                                  | Where your Earth Engine export goes (not committed to git)                                                      |
| `data/tifs/`                                   | Small sample tiles from the original Galileo repository                                                         |
| `src/`, `config/`, `*.py` in the root          | The Galileo source code: model, data pipeline, training and evaluation                                          |
| `src/data/copernicus/`, `copernicus_marimo.py` | A pipeline for Sentinel-1/2 data from the Copernicus Data Space                                                 |
| `docs/GALILEO_UPSTREAM_README.md`              | The original Galileo README (model sizes, training, Docker, tests)                                              |

Everything participants generate (`outputs/`, `data/busia/*.tif`, the CropHarvest download in
`data/cropharvest_data/`) is excluded from git.

## Web MVP

`mvp/` is a shippable demo built on the case study: a Leaflet map with place search, polygon
drawing and an analysis period. It returns maize probability, acreage, confidence and an
insurance proxy, caches every analysis, and shows change over time for the same area.

- Cached Busia 2019 tile: instant.
- Any other area/period: on-demand Sentinel export from Earth Engine, then frozen Galileo +
  the saved maize classifier.
- Run: `cd mvp && cp .env.example .env` (set `EE_PROJECT` and `EE_BUCKET`), then
  `docker compose up --build`, open http://localhost:8000.
- Details: [mvp/README.md](mvp/README.md).

## About Galileo

Galileo is a family of pretrained remote sensing models developed by NASA Harvest, Ai2, McGill
University and partners, published at ICML 2025. It learns from Sentinel-1 and Sentinel-2 time series,
elevation, weather, land-cover maps and other inputs, and handles both images and pixel time series.

- Original repository: https://github.com/nasaharvest/galileo
- Paper: https://arxiv.org/abs/2502.09356
- Other model sizes (tiny, base): https://huggingface.co/nasaharvest/galileo

If you use Galileo, please cite the authors:

```
@misc{tseng2025galileolearninggloballocal,
      title={Galileo: Learning Global and Local Features in Pretrained Remote Sensing Models},
      author={Gabriel Tseng and Anthony Fuller and Marlena Reil and Henry Herzog and Patrick Beukema and Favyen Bastani and James R. Green and Evan Shelhamer and Hannah Kerner and David Rolnick},
      year={2025},
      eprint={2502.09356},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.09356},
}
```

The case study also uses the **CropHarvest** dataset (Tseng et al., 2021;
https://zenodo.org/record/10251170) for the Kenya maize labels.

## Licence

Released under the MIT licence, as in the original Galileo repository (see [LICENSE](LICENSE)).
The case-study materials in `case_study/` are shared under the same licence. CropHarvest and
the satellite data used in the exercises have their own terms.
