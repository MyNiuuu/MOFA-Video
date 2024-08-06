## Introduction

This repo provides the inference Script for **Keypoint-Based** Control of MOFA-Video that supports **long video generation** via the proposed periodic sampling strategy.

## Get Started

### 1. Clone the Repository

```
git clone https://github.com/MyNiuuu/MOFA-Video.git
cd ./MOFA-Video
```

### 2. Environment Setup

This script has been tested on CUDA version of 11.7.

```
cd ./MOFA-Video-Keypoint
conda create -n mofa_ldmk python==3.10
conda activate mofa_ldmk
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
cd ..
```

#### 3. Downloading Checkpoints

1. Download the checkpoint of CMP from [here](https://huggingface.co/MyNiuuu/MOFA-Video-Hybrid/resolve/main/models/cmp/experiments/semiauto_annot/resnet50_vip%2Bmpii_liteflow/checkpoints/ckpt_iter_42000.pth.tar) and put it into `./MOFA-Video-Keypoint/models/cmp/experiments/semiauto_annot/resnet50_vip+mpii_liteflow/checkpoints`.

2. Download the `ckpts` [folder](https://huggingface.co/MyNiuuu/MOFA-Video-Hybrid/tree/main/ckpts) from the huggingface repo which contains necessary pretrained checkpoints and put it under `./MOFA-Video-Keypoint`. You may use `git lfs` to download the **entire** `ckpts` folder:

    1) Download `git lfs` from https://git-lfs.github.com. It is commonly used for cloning repositories with large model checkpoints on HuggingFace.

        **NOTE:** If you encounter the error `git: 'lfs' is not a git command` on Linux, you can try [this solution](https://github.com/text2cinemagraph/text2cinemagraph/issues/1) that has worked well for my case.

    2) Execute `git clone https://huggingface.co/MyNiuuu/MOFA-Video-Hybrid` to download the complete HuggingFace repository, which includes the `ckpts` folder.
    3) Copy or move the `ckpts` folder to `./MOFA-Video-Keypoint`.

    

#### 3. Running Inference Scripts

```
cd ./MOFA-Video-Keypoint
chmod 777 inference.sh
./inference.sh
```
