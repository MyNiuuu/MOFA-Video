# MOFA-Video: Controllable Image Animation via Generative Motion Field Adaptions in Frozen Image-to-Video Diffusion Model.

[Muyao Niu](https://myniuuu.github.io/), 
[Xiaodong Cun](https://vinthony.github.io/academic/), 
[Xintao Wang](https://xinntao.github.io/), 
[Yong Zhang](https://yzhang2016.github.io/), 
[Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en), 
[Yinqiang Zheng](https://scholar.google.com/citations?user=JD-5DKcAAAAJ&hl=en)

[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)](https://myniuuu.github.io/MOFA_Video)


## Introduction
<p align="center">
  <img src="assets/figures/project-mofa.png">
</p>
We introduce MOFA-Video, a method designed to adapt motions from different domains to the frozen Video Diffusion Model. By employing <u>sparse-to-dense (S2D) motion generation</u> and <u>flow-based motion adaptation</u>, MOFA-Video can effectively animate a single image using various types of control signals, including trajectories, keypoint sequences, AND their combinations.
<p align="center">
  <img src="assets/figures/pipeline.png">
</p>
During the training stage, we generate sparse control signals through sparse motion sampling and then train different MOFA-Adapters to generate video via pre-trained SVD. During the inference stage, different MOFA-Adapters can be combined to jointly control the frozen SVD.

---

please check the gallery of our [project page](https://myniuuu.github.io/MOFA_Video) for many visual results!

## 📰 **TODO**
- [ ] Gradio demo and checkpoints for trajectory-based image animation (By this weekend)
- [ ] Inference scripts and checkpoints for keypoint-based facial image animation
- [ ] inference Gradio demo for hybrid image animation
- [ ] Training codes 


# Acknowledgements
We appreciate the Gradio code of [DragNUWA](https://arxiv.org/abs/2308.08089).
