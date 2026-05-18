# HAAF

[![Paper](https://img.shields.io/badge/Paper-ACM%20DL-blue)](https://dl.acm.org/doi/10.1145/3774904.3793015)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3774904.3793015-green)](https://doi.org/10.1145/3774904.3793015)
[![Python](https://img.shields.io/badge/Python-3.8-lightgrey)](#environment)

Official code release for **HAAF: Hierarchical Adaptation and Alignment of Foundation Models for Few-Shot Pathology Anomaly Detection**.

**Paper:** [ACM Digital Library](https://dl.acm.org/doi/10.1145/3774904.3793015) | [DOI](https://doi.org/10.1145/3774904.3793015) | [arXiv](https://arxiv.org/abs/2601.17405)

HAAF addresses few-shot pathology anomaly detection by adapting frozen pathology foundation models with lightweight, hierarchy-aware vision-language alignment. The released code includes the HAAF training/evaluation entry, recovered local adapter dependency, fixed few-shot support splits, and checkpoint preparation utilities.

## Highlights

- **Foundation-model backbone:** CONCH v1.5 visual patch features with HAAF detection adapters.
- **Adaptive pathology prompts:** learnable normal/abnormal prompts for each pathology dataset.
- **Hierarchical alignment:** sparse MMA-style adapters and cross-attention align text and visual representations across layers.
- **Stable few-shot inference:** dual scoring from text-guided anomaly probabilities and support-set prototypes.

## Contents

- [Environment](#environment)
- [Checkpoints](#checkpoints)
- [Datasets](#datasets)
- [Few-Shot Splits](#few-shot-splits)
- [Run HAAF](#run-haaf)
- [Repository Layout](#repository-layout)
- [Citation](#citation)

## Environment

On the 93 server, use the existing conda environment:

```bash
conda activate mvfa
```

For a fresh machine, create an equivalent environment named `mvfa`:

```bash
conda create -n mvfa python=3.8 -y
conda activate mvfa
pip install -r requirements.txt
```

The recovered server environment used Python 3.8 and PyTorch 1.11.0+cu113. Install the PyTorch wheel that matches your CUDA driver if needed.

## Checkpoints

Model weights are not redistributed in this repository.

Prepare OpenAI CLIP files:

```bash
python scripts/prepare_checkpoints.py --clip
```

Expected CLIP files:

```text
CLIP/bpe_simple_vocab_16e6.txt.gz
CLIP/ckpt/ViT-L-14-336px.pt
```

Prepare CONCH v1.5 after your Hugging Face account has access to the gated `MahmoodLab/conchv1_5` repository:

```bash
huggingface-cli login
python scripts/prepare_checkpoints.py --conch
```

Expected CONCH file:

```text
conch/checkpoints/pytorch_model_vision.bin
```

## Datasets

Raw medical datasets are not redistributed in this repository. Download the source datasets, then arrange the processed anomaly-detection folders under `--data_path` in the `*_AD` format shown below.

| `--obj` | Source dataset | Public access |
| --- | --- | --- |
| `Histopathology` | BreaKHis breast cancer histopathology images | [Download page](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) |
| `CRC` | CRC-VAL-HE-7K / NCT-CRC-HE colorectal histology | [Zenodo](https://zenodo.org/records/1214456) |
| `SICAP` | SICAPv2 prostate whole-slide image patches | [Mendeley Data](https://data.mendeley.com/datasets/9xxm58dvs3/1) |
| `BRACS` | BRACS breast carcinoma subtyping dataset | [Official site](https://www.bracs.icar.cnr.it/) |

Expected processed layout:

```text
data/
  Histopathology_AD/
    valid/good/img/
    valid/Ungood/img/
    valid/Ungood/anomaly_mask/      # optional for image-level pathology datasets
    test/good/img/
    test/Ungood/img/
    test/Ungood/anomaly_mask/       # optional for image-level pathology datasets
  CRC_AD/
  SICAP_AD/
  BRACS_AD/
```

For reproducibility, the processed paper splits used the following image counts:

| Dataset | valid good | valid abnormal | test good | test abnormal |
| --- | ---: | ---: | ---: | ---: |
| Histopathology | 117 | 119 | 1003 | 994 |
| CRC | 16 | 16 | 4324 | 1973 |
| SICAP | 100 | 100 | 1000 | 1000 |
| BRACS | 242 | 328 | 1460 | 2197 |

The loader also contains legacy support for `Brain`, `Liver`, `Retina_RESC`, `Retina_OCT2017`, and `Chest`, but the pathology release is centered on `Histopathology`, `CRC`, `SICAP`, and `BRACS`.

## Few-Shot Splits

Fixed support-image splits are included in:

```text
dataset/fewshot_seed/<OBJ>/<SHOT>-shot.txt
```

Released pathology splits:

| Dataset | Released shots |
| --- | --- |
| `Histopathology` | 2, 4, 8, 16 |
| `CRC` | 2, 4, 8 |
| `SICAP` | 2, 4, 8, 16 |
| `BRACS` | 2, 4, 8, 16 |

Each split file contains normal and abnormal support filenames:

```text
a-0: abnormal_1.png abnormal_2.png abnormal_3.png abnormal_4.png
n-0: normal_1.png normal_2.png normal_3.png normal_4.png
```

Use `--iterate 0` to reproduce the released split. Use `--iterate -1` to sample support images randomly.

## Run HAAF

Example 4-shot HAAF run:

```bash
python train_CLAS.py \
  --obj Histopathology \
  --data_path ./data \
  --device cuda:0 \
  --shot 4 \
  --epoch 50 \
  --iterate 0
```

Run the four pathology datasets:

```bash
for obj in Histopathology CRC SICAP BRACS; do
  python train_CLAS.py \
    --obj "$obj" \
    --data_path ./data \
    --device cuda:0 \
    --shot 4 \
    --epoch 50 \
    --iterate 0
done
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--disable-mma` | Ablation without HAAF MMA cross-attention. |
| `--coop-n-ctx 16` | Number of learnable prompt context tokens. |
| `--mma-joint-layers 3 6 9 12` | Text layers used for cross-level alignment. |
| `--features_list 6 12 18 24` | CONCH visual layers used for patch features. |

Outputs are written to:

```text
result/FINAL/<OBJ>.txt
ckpt/few_mma_gemini/<OBJ>.pth
```

## Repository Layout

```text
HAAF/
  train_CLAS.py                  # main HAAF training/evaluation entry
  dataset/medical_few.py         # dataset loader and fixed few-shot support splits
  dataset/fewshot_seed/          # released support-image splits used for reproduction
  MultiModalAdapter/             # local MMA/CoOp adapter dependency
  CLIP/                          # vendored CLIP wrapper used by HAAF
  conch/open_clip_custom/        # CONCH detection adapter wrapper
  loss.py, utils.py, prompt.py   # losses, augmentation, class text names
  scripts/prepare_checkpoints.py # helper for external weights
```

Large checkpoints, datasets, run logs, and result folders are intentionally excluded from git.

## Notes

- Keep `--batch_size 1`; the current prototype-building and evaluation path assumes one image per batch.
- HAAF is the default configuration. `--disable-mma` is an ablation switch, not the main method.
- Backbone weights are frozen. Training updates CONCH detection adapters, prompt tokens, MMA bottleneck adapters, cross-attention modules, and the shared logit scale.
- CONCH v1.5 weights are gated and must not be redistributed in this repository.

## Citation

If you find HAAF useful, please cite the ACM version:

```bibtex
@inproceedings{yang2026haaf,
  title     = {HAAF: Hierarchical Adaptation and Alignment of Foundation Models for Few-Shot Pathology Anomaly Detection},
  author    = {Yang, Chunze and Zhao, Wenjie and Tang, Yue and Lu, Junbo and Ge, Jiusong and Liu, Qidong and Gao, Zeyu and Li, Chen},
  booktitle = {Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3774904.3793015},
  url       = {https://doi.org/10.1145/3774904.3793015}
}
```
