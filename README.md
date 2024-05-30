



<div align="center">
  <h2>
    MOFA-Video: Controllable Image Animation via Generative Motion Field Adaptions <br> in Frozen Image-to-Video Diffusion Model
  </h2>
<a href=''><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp; <a href='https://myniuuu.github.io/MOFA_Video'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<div>
    <a href='https://myniuuu.github.io/' target='_blank'>Muyao Niu</a> <sup>1,2</sup> &nbsp;
    <a href='https://vinthony.github.io/academic/' target='_blank'>Xiaodong Cun</a><sup>2,*</sup> &nbsp;
    <a href='https://xinntao.github.io/' target='_blank'>Xintao Wang</a><sup>2</sup> &nbsp;
    <a href='https://yzhang2016.github.io/' target='_blank'>Yong Zhang</a><sup>2</sup> &nbsp; <br>
    <a href='https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en' target='_blank'>Ying Shan</a><sup>2</sup> &nbsp;
    <a href='https://scholar.google.com/citations?user=JD-5DKcAAAAJ&hl=en' target='_blank'>Yinqiang Zheng</a><sup>1</sup> &nbsp;
</div>
<div>
    <sup>1</sup> The University of Tokyo &nbsp; <sup>2</sup> Tencent AI Lab &nbsp; <sup>*</sup> Corresponding Author &nbsp; 
</div>
</div>


## Introduction
<p align="center">
  <img src="assets/images/project-mofa.png">
</p>
We introduce MOFA-Video, a method designed to adapt motions from different domains to the frozen Video Diffusion Model. By employing <u>sparse-to-dense (S2D) motion generation</u> and <u>flow-based motion adaptation</u>, MOFA-Video can effectively animate a single image using various types of control signals, including trajectories, keypoint sequences, AND their combinations.
<br>
<br>
<p align="center">
  <img src="assets/images/pipeline.png">
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
