## Introduction

This is the guidance for training your own MOFA-Adapter.

🔧🔧🔧 Stay tuned. There may exist bugs, feel free to contact me or report issues!

## Environment Setup
```
conda create -n mofa_train python==3.10
conda activate mofa_train
pip install -r requirements.txt
```

## Datset Preparation

We train our MOFA-Adapter on WebVid-10M. Please refer to our implementation of `WebVid10M` class in `./train_utils/dataset.py` for more details about how to read the data. You may need to download the WebVid-10M first, or you can modify the codes of `WebVid10M` class and train your own MOFA-Adapter on other datasets.

## Download checkpoints

1. Download the pretrained checkpoint folder of [SVD_xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) from huggingface to `./ckpts`.

    The structure of the checkpoint folder should be:
    ```
    ./ckpts
    |-- stable-video-diffusion-img2vid-xt-1-1
    |   |-- feature_extractor
    |       |-- ...
    |   |-- image_encoder
    |       |-- ...
    |   |-- scheduler
    |       |-- ...
    |   |-- unet
    |       |-- ...
    |   |-- vae
    |       |-- ...
    |   |-- svd_xt_1_1.safetensors
    |   `-- model_index.json
    ```

2. Download the Unimatch checkpoint from [here](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth) and put it into `./train_utils/unimatch/pretrained`.

3. Download the checkpoint of CMP from [here](https://huggingface.co/MyNiuuu/MOFA-Video-Traj/blob/main/models/cmp/experiments/semiauto_annot/resnet50_vip%2Bmpii_liteflow/checkpoints/ckpt_iter_42000.pth.tar) and put it into `./models/cmp/experiments/semiauto_annot/resnet50_vip+mpii_liteflow/checkpoints`.


## Run Training scripts

### Stage 1

```
./train_stage1.sh
```


### Stage 2

Change the value of `--controlnet_model_name_or_path` in `train_stage2.sh`, then run:

```
./train_stage2.sh
```