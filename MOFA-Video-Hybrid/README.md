
## Introduction

This repo provides the inference Gradio demo for **Hybrid (Trajectory + Landmark)** Control of MOFA-Video.

## Environment Setup

```
cd MOFA-Hybrid
conda create -n mofa python==3.10
conda activate mofa
pip install -r requirements.txt
pip install opencv-python-headless
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

**IMPORTANT:** ⚠️⚠️⚠️ Gradio Version of **4.5.0** should be used since other versions may cause errors.


## Checkpoints Download
1. Download the checkpoint of CMP from [here](https://huggingface.co/MyNiuuu/MOFA-Video-Hybrid/resolve/main/models/cmp/experiments/semiauto_annot/resnet50_vip%2Bmpii_liteflow/checkpoints/ckpt_iter_42000.pth.tar) and put it into `./models/cmp/experiments/semiauto_annot/resnet50_vip+mpii_liteflow/checkpoints`.

2. Downloading the necessary pretrained checkpoints from [huggingface](https://huggingface.co/MyNiuuu/MOFA-Video-Hybrid). It is recommended to directly using git lfs to clone the [huggingface repo](https://huggingface.co/MyNiuuu/MOFA-Video-Hybrid). The checkpoints should be orgnized as `./ckpt_tree.md` (they will be automatically organized if you use git lfs to clone the [huggingface repo](https://huggingface.co/MyNiuuu/MOFA-Video-Hybrid)).


## Run Gradio Demo

### Using audio to animate the facial part

`python run_gradio_audio_driven.py`

### Using refernce video to animate the facial part

`python run_gradio_video_driven.py`

**IMPORTANT:** ⚠️⚠️⚠️ Please refer to the instructions on the gradio interface during the inference process!


## Acknowledgements
We use [SadTalker](https://github.com/OpenTalker/SadTalker) and [AniPortrait](https://github.com/Zejun-Yang/AniPortrait) to generate the landmarks in this demo. We sincerely appreciate their code and checkpoint release.

