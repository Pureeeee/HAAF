# HAAF

Official code release for **HAAF: Hierarchical Adaptation and Alignment of Foundation Models for Few-Shot Pathology Anomaly Detection**.

HAAF is the default method implemented in this repository. It combines frozen pathology foundation-model features with lightweight adaptation modules for few-shot pathology anomaly detection:

- CONCH v1.5 visual patch features and HAAF detection adapters.
- Learnable normal/abnormal pathology prompts.
- Hierarchical vision-text alignment with sparse MMA adapters and cross-attention.
- Prototype scoring from few-shot support samples plus text-guided anomaly scoring.

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

## Environment

On the 93 server, the code is intended to run in the existing conda environment:

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

Raw medical datasets are not redistributed in this repository. Download the source datasets, then arrange the processed anomaly-detection folders under `--data_path` in the `*_AD` format below.

| `--obj` | Source dataset | Public access |
| --- | --- | --- |
| `Histopathology` | BreaKHis breast cancer histopathology images | https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/ |
| `CRC` | CRC-VAL-HE-7K / NCT-CRC-HE colorectal histology | https://zenodo.org/records/1214456 |
| `SICAP` | SICAPv2 prostate whole-slide image patches | https://data.mendeley.com/datasets/9xxm58dvs3/1 |
| `BRACS` | BRACS breast carcinoma subtyping dataset | https://www.bracs.icar.cnr.it/ |

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

```text
Histopathology: 2, 4, 8, 16 shots
CRC:            2, 4, 8 shots
SICAP:          2, 4, 8, 16 shots
BRACS:          2, 4, 8, 16 shots
```

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

```text
--disable-mma              ablation without HAAF MMA cross-attention
--coop-n-ctx 16            number of learnable prompt context tokens
--mma-joint-layers 3 6 9 12
--features_list 6 12 18 24
```

Outputs are written to:

```text
result/FINAL/<OBJ>.txt
ckpt/few_mma_gemini/<OBJ>.pth
```

## Notes

- Keep `--batch_size 1`; the current prototype-building and evaluation path assumes one image per batch.
- HAAF is the default configuration. `--disable-mma` is an ablation switch, not the main method.
- Backbone weights are frozen. Training updates CONCH detection adapters, prompt tokens, MMA bottleneck adapters, cross-attention modules, and the shared logit scale.
- CONCH v1.5 weights are gated and must not be redistributed in this repository.
