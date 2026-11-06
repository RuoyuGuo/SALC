<div align="center">

# Sparse All-Layer Connector for Domain Generalised Semantic Segmentation

Ruoyu Guo* · Xin Kun Lin* · Maurice Pagnucco · Yang Song

School of Computer Science and Engineering,<br>
University of New South Wales

<sup>*</sup>Equal contribution.

**NeurIPS 2026**

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2026-blue)](https://arxiv.org/abs/2610.15660)
[![arXiv](https://img.shields.io/badge/arXiv-2610.15660-b31b1b.svg)](https://arxiv.org/abs/2610.15660)
[![Weights](https://img.shields.io/badge/Weights-SALC-green)](#checkpoints)

</div>

Official implementation of the Neurips2026 paper **Sparse All-Layer Connector for Domain Generalised Semantic Segmentation**.


## Installation

The environment setup follows [HPI](https://github.com/MosukFate/HPI): Python 3.10, PyTorch 2.0.1, CUDA 11.8, and MMCV-full 1.7.2.

```bash
conda create -n salc python=3.10 -y
conda activate salc
pip install -r requirements.txt
pip install causal-conv1d==1.4.0 mamba-ssm==2.2.2 --no-build-isolation
```

## Pre-trained VFM & VLM Models

Download the pre-trained models and save them in `./pretrained/`:

| Model | Filename | Download |
| --- | --- | --- |
| DINOv2 | `dinov2_vitl14_pretrain.pth` | [Download](https://drive.google.com/file/d/1Rrl0RfU51eU8orbNVWHtNr1L3k5xhnld/view?usp=sharing) |
| CLIP | `ViT-L-14-336px.pt` | [Download](https://drive.google.com/file/d/1s00ofvxn0NCVVgnycXd2wUx4Gs53O6mj/view?usp=sharing) |
| SigLIP | `siglip_vitl16_384.pth` | [Download](https://drive.google.com/file/d/1PezEbwpqlasSYH2KPtU3aUD4hCzk9uE-/view?usp=sharing) |


```text
SALC/
└── pretrained/
    ├── dinov2_vitl14_pretrain.pth
    ├── ViT-L-14-336px.pt
    └── siglip_vitl16_384.pth
```

## Checkpoints

Checkpoint download links are to be added. Save each checkpoint in `./work_dirs_d/` using the filename listed below. One checkpoint is shared across the evaluation targets for the same model and training domain.

| Filename | Model combination | Trained on | Download |
| --- | --- | --- | --- |
| `clip-gta.pth` | CLIP + DINOv2 | GTA | [Download][ckpt-clip-gta] |
| `clip-syn.pth` | CLIP + DINOv2 | SYNTHIA | [...][ckpt-clip-syn] |
| `clip-city.pth` | CLIP + DINOv2 | Cityscapes | [...][ckpt-clip-city] |
| `siglip-gta.pth` | SigLIP + DINOv2 | GTA | [Download][ckpt-siglip-gta] |
| `siglip-syn.pth` | SigLIP + DINOv2 | SYNTHIA | [...][ckpt-siglip-syn] |
| `siglip-city.pth` | SigLIP + DINOv2 | Cityscapes | [...][ckpt-siglip-city] |

## Datasets

- Please refer to [TLDR](https://github.com/ssssshwan/TLDR#setup-datasets) for dataset download links.

- After downloading the datasets, edit the data configs in `/SALC/configs/_base_/datasets` following your environment.

  ```python
  src_dataset_dict = dict(..., data_root='[YOUR_DATA_FOLDER_ROOT]', ...)
  tgt_dataset_dict = dict(..., data_root='[YOUR_DATA_FOLDER_ROOT]', ...)
  ```

- **Data Preprocessing:** Edit the dataset paths and enable the required conversions in `tools/convert_datasets/convert.sh`, then run:

    ```shell
    (cd tools/convert_datasets && sh convert.sh)
    ```

- Note that RAND_CITYSCAPES is SYNTHIA. The final folder structure should look like this:

```
.
├── SALC
│   ├── ...
│   ├── pretrained
│   │   ├── dinov2_vitl14_pretrain.pth
│   │   ├── siglip_vitl16_384.pth
│   │   └── ViT-L-14-336px.pt
│   └── ...
└── dataset
    ├── cityscapes
    │   ├── leftImg8bit
    │   │   ├── train
    │   │   └── val
    │   └── gtFine
    │       ├── train
    │       └── val
    ├── bdd100k
    │   ├── images
    │   │   └── 10k
    │   │       └── val
    │   └── labels
    │       └── sem_seg
    │           └── masks
    │               └── val
    ├── mapillary
    │   └── half
    │       ├── images
    │       └── labels
    ├── gta
    │   ├── images
    │   └── labels
    └── RAND_CITYSCAPES
        ├── RGB
        └── GT
            └── LABELS
```

## Configurations

Following the GI-Adapter layout, each configuration includes both its source training dataset and target evaluation dataset.

| Training domain | Evaluation target | CLIP + DINOv2 | SigLIP + DINOv2 |
| --- | --- | --- | --- |
| GTA | Cityscapes | [clip_g2c.py](configs/g2cbm/clip_g2c.py) | [siglip_g2c.py](configs/g2cbm/siglip_g2c.py) |
| GTA | BDD100K | [clip_g2b.py](configs/g2cbm/clip_g2b.py) | [siglip_g2b.py](configs/g2cbm/siglip_g2b.py) |
| GTA | Mapillary | [clip_g2m.py](configs/g2cbm/clip_g2m.py) | [siglip_g2m.py](configs/g2cbm/siglip_g2m.py) |
| SYNTHIA | Cityscapes | [clip_s2c.py](configs/s2cbm/clip_s2c.py) | [siglip_s2c.py](configs/s2cbm/siglip_s2c.py) |
| SYNTHIA | BDD100K | [clip_s2b.py](configs/s2cbm/clip_s2b.py) | [siglip_s2b.py](configs/s2cbm/siglip_s2b.py) |
| SYNTHIA | Mapillary | [clip_s2m.py](configs/s2cbm/clip_s2m.py) | [siglip_s2m.py](configs/s2cbm/siglip_s2m.py) |
| Cityscapes | BDD100K | [clip_c2b.py](configs/c2bm/clip_c2b.py) | [siglip_c2b.py](configs/c2bm/siglip_c2b.py) |
| Cityscapes | Mapillary | [clip_c2m.py](configs/c2bm/clip_c2m.py) | [siglip_c2m.py](configs/c2bm/siglip_c2m.py) |


## Training

```
python train.py configs/[TRAIN_CONFIG]
```

## Evaluation

```
python test.py configs/[TEST_CONFIG] work_dirs_d/[MODEL-NAME] --eval mIoU
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{guo2026salc,
  title={Sparse All-Layer Connector for Domain Generalised Semantic Segmentation},
  author={Guo, Ruoyu and Lin, Xin Kun and Pagnucco, Maurice and Song, Yang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  url={https://arxiv.org/abs/2610.15660}
}
```

## Acknowledgements

This project builds on [MFuser](https://github.com/devinxzhang/MFuser).

<!-- Replace the six CKPT_URL_* placeholders below with public download URLs. -->
[ckpt-clip-gta]: https://huggingface.co/ryasdf/SALC/tree/main
[ckpt-clip-syn]: CKPT_URL_CLIP_SYN
[ckpt-clip-city]: CKPT_URL_CLIP_CITY
[ckpt-siglip-gta]: https://huggingface.co/ryasdf/SALC/tree/main
[ckpt-siglip-syn]: CKPT_URL_SIGLIP_SYN
[ckpt-siglip-city]: CKPT_URL_SIGLIP_CITY
