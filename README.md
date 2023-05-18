# FS-Tox: An *In Vivo* Toxicity Benchmark
![License](https://img.shields.io/github/license/sethhowes/fs-tox)
![Python Version](https://img.shields.io/badge/python-3.7-blue)
![Project Status](https://img.shields.io/badge/status-alpha-red)

## 🔎 Overview
We are building FS-Tox: a toxicity benchmark for *in vivo* small molecule toxicology assays. Toxicity prediction tasks differ from traditional machine learning tasks in that there are usually only a small number of training examples per toxicity assay. Here, we are creating a benchmarking tool built using several publicly available toxicity datasets (e.g. EPA's ToxRefDB). We will incorporate the different in vivo assays from these datsets consisting of the molecular representation of a small molecule, with an associated binary marker of whether the drug was toxic or not for the given assay.

## 🗺️ Roadmap
### Mid-May 2023 - benchmark SOTA models
We will test the performance of the following state-of-the-art few-shot prediction methods: 
- [x] Gradient Boosted Random Forest on ECFP4 fingerprints
- [] Text-embedding-ada-002 on SMILES (OpenAI)
- [] Galactica 125M (Hugging Face)
- [] Galactica 1.3B (Hugging Face)
- [] ChemGPT 19M (Hugging Face)
- [] ChemGPT 1.2B (Hugging Face)
- [] Uni-Mol (docker)
- [] Uni-Mol+ (docker)
- [] MoLeR (Microsoft)

### Late-May 2023 - create FS-Tox benchmarking tool
We will incorporate the following datsets containing results from *in vivo* toxicity assays:
- [] ToxRefDB (subacute and chronic toxicity)
- [] TDCommon, Zhu 2009 (acute toxicity)
- [] MEIC (small, curated clinical toxicity)

### Early-June 2023 - benchmark SOTA small molecule language models on FS-Tox
We will test the following language models on the FS-Tox benchmark:
- [] Text-embedding-ada-002 on SMILES (OpenAI)
- [] Galactica 125M (Hugging Face)
- [] Galactica 1.3B (Hugging Face)
- [] ChemGPT 19M (Hugging Face)
- [] ChemGPT 1.2B (Hugging Face)
- [] Uni-Mol (docker)
- [] Uni-Mol+ (docker)
- [] MoLeR (Microsoft)

### Mid-June 2023 - extend FS-Tox with *in vitro* data
Incorporate *in vitro* assays into the FS-Tox benchmark:
- [] ToxCast
- [] Extended Tox21
- [] NCI60 data

## 📂 Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         and a short `-` delimited description, e.g.
    │                         `1.0-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

## 📚 Resources
1. [ToxRefDB version 2.0: Improved utility for predictive and retrospective toxicology analyses](https://pubmed.ncbi.nlm.nih.gov/31340180/)
2. [ChemGPT: a transformer model for generative molecular modeling](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/627bddd544bdd532395fb4b5/original/neural-scaling-of-deep-chemical-models.pdf) 