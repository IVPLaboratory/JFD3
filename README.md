# JFD3
Official code for the AAAI 2026 paper “Blur-Robust Detection via Feature Restoration: An End-to-End Framework for Prior-Guided Infrared UAV Target Detection”[![arXiv](https://img.shields.io/badge/arXiv-2511.14371-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2511.14371)

## Blur-Robust Detection via Feature Restoration: An End-to-End Framework for Prior-Guided Infrared UAV Target Detection ##

Authors: Xiaolin Wang<sup>1</sup>, Houzhang Fang<sup>1</sup>, Qingshan Li<sup>1</sup>, Lu Wang<sup>1</sup>, Yi Chang<sup>2</sup>, Luxin Yan<sup>2</sup>
  
<sup>1</sup>Xidian University,  <sup>2</sup>Huazhong University of Science and Technology


🎯 We have first compiled the relevant code for the core contribution points, and the complete code is currently being compiled. We will update the supplementary materials and code in the future.

## Abstract
Infrared unmanned aerial vehicle (UAV) target images often suffer from motion blur degradation caused by rapid sensor movement, significantly reducing contrast between target and background. Generally, detection performance heavily depends on the discriminative feature representation between target and background. Existing methods typically treat deblurring as a preprocessing step focused on visual quality, while neglecting the enhancement of task-relevant features crucial for detection. Improving feature representation for detection under blur conditions remains challenging. In this paper, we propose a novel **J**oint **F**eature-**D**omain **D**eblurring and **D**etection  end-to-end framework, dubbed JFD<sup>3</sup>. We design a dual-branch architecture with shared weights, where the clear branch guides the blurred branch to enhance discriminative feature representation. Specifically, we first introduce a lightweight feature restoration network, where features from the clear branch serve as feature-level supervision to guide the blurred branch, thereby enhancing its distinctive capability for detection. %act as feature-level supervision to constrain the blurred branch to improve the distinctive capability for detection. We then propose a frequency structure guidance module that refines the structure prior from the restoration network and integrates it into shallow detection layers to enrich target structural information. Finally, a feature consistency self-supervised loss is imposed between the dual-branch detection backbones, driving the blurred branch to approximate the feature representations of the clear one. We also construct a benchmark, named IRBlurUAV, containing 30,000 simulated and 4,118 real infrared UAV target images with diverse motion blur. Extensive experiments on IRBlurUAV demonstrate that JFD<sup>3</sup> achieves superior detection performance while maintaining real-time efficiency. 



## Network Architecture
<p align="center">
  <img src="figs/JFD3.png" width="auto" alt="accessibility text">
</p>
Overall architecture of the proposed JFD\textsuperscript{3}.

## Dataset Download Link

The IRBlurUAV dataset used in our paper can be downloaded via the following link:

- Download Dataset Here -[Baidu Netdisk](<https://pan.baidu.com/s/170bAY4mJkAqbCf4ythljvA?pwd=xpuv>),    
                        -[Google Drive(TODO)](<>)

We developed the IRBlurUAV dataset, which involved considerable effort. If you use this dataset in your work, please cite our contribution. Thank you!





## 🔗 Citation
If you find our work and our dataset IRBlurUAV useful for your research, please consider citing our paper. Thank you!
```
@inproceedings{2026AAAI_JFD3,
  title     = {Blur-Robust Detection via Feature Restoration: An End-to-End Framework for Prior-Guided Infrared UAV Target Detection},
  author    = {Xiaolin Wang and Houzhang Fang and Qingshan Li and Lu Wang and Yi Chang and Luxin Yan},
  booktitle   = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
  pages     = { },
}
```

In additoin to the above paper, please also consider citing the following references. Thank you!
```

@inproceedings{2025CVPR_UniCD,
    title     = {Detection-Friendly Nonuniformity Correction: A Union Framework for Infrared {UAV} Target Detection},
    author    = {Houzhang Fang and Xiaolin Wang and Zengyang Li and Lu Wang and Qingshan Li and Yi Chang and Luxin Yan},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2025},
    month     =  {June},
    pages     = {11898-11907},
}

@article{2024TGRS_SCINet,
  title     = {{SCINet}: Spatial and Contrast Interactive Super-Resolution Assisted Infrared {UAV} Target Detection},
  author    = {Houzhang Fang and Lan Ding and Xiaolin Wang and Yi Chang and Luxin Yan and Li Liu and Jinrui Fang},
  journal   = {IEEE Transactions on Geoscience and Remote Sensing},
  volume    = {62},
  year      = {2024},
  pages     = {1-22},
}
@ARTICLE{2023TII_DAGNet,
  title     =  {Differentiated Attention Guided Network Over Hierarchical and Aggregated Features for Intelligent {UAV} Surveillance},
  author    =  {Houzhang Fang and Zikai Liao and Xuhua Wang and Yi Chang and Luxin Yan},
  journal   =  {IEEE Transactions on Industrial Informatics}, 
  year      =  {2023},
  volume    =  {19},
  number    =  {9},
  pages     =  {9909-9920},
  }
@inproceedings{2023ACMMM_DANet,
title       =  {{DANet}: Multi-scale {UAV} Target Detection with Dynamic Feature Perception and Scale-aware Knowledge Distillation},
author      =  {Houzhang Fang and Zikai Liao and Lu Wang and Qingshan Li and Yi Chang and Luxin Yan and Xuhua Wang},
booktitle   =  {Proceedings of the 31st ACM International Conference on Multimedia (ACMMM)},
pages       =  {2121-2130},
year        =  {2023},
}
@ARTICLE{2022TIMFang,
  title     =  {Infrared Small {UAV} Target Detection Based on Depthwise Separable Residual Dense Network and Multiscale Feature Fusion},
  author    =  {Houzhang Fang and Lan Ding and Liming Wang and Yi Chang and Luxin Yan and Jinhui Han},
  journal   =  {IEEE Transactions on Instrumentation and Measurement}, 
  year      =  {2022},
  volume    =  {71},
  number    =  {},
  pages     =  {1-20},
}
```

## 🤝Acknowledgement

Our implementation is built upon the [DEIM](<https://github.com/Intellindust-AI-Lab/DEIM>) codebase. For parts that are not the core contributions of our paper, we refer the readers to the original repository

