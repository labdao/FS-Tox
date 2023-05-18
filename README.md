# FS-Tox: An *In Vivo* Toxicity Benchmark

## Overview
We are building FS-Tox: a toxicity benchmark for *in vivo* toxicology assays. Toxicity prediction tasks differ from traditional machine learning tasks in that there are commonly only a small number of training examples for a given toxicity assay. Here, we are creating a benchmarking tool built using several publicly available toxicity datasets (e.g. EPA's ToxRefDB). We will incorporate the different in vivo assays from these datsets consisting of the molecular representation of a small molecule, with an associated binary marker of whether the drug was toxic or not for the given assay.

## Roadmap
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

## Resources
Here are some resources to take a look at: