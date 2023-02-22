# Official repository for "Exploring Self-Supervised Vision Transformers for Gait Recognition in the Wild"

## Adrian Cosma, Andy Catruna, Emilian Radoi

### Abstract

*The manner of walking (gait) can be a powerful biometric that can be used as a unique fingerprinting method, allowing unobtrusive behavioural analytics to be performed at a distance without explicit subject cooperation. As opposed to more traditional biometric authentication methods, gait analysis does not require cooperation of the subject and can be performed in low resolution settings, without requiring the subject's face to be clearly visible and unobstructed.
Most current approaches are developed in a controlled setting, with clean, gold-standard annotated data, which powered the development of neural architectures for recognition and classification. Only recently has gait analysis ventured into using more diverse, large-scale and realistic datasets to pretrain networks in a self-supervised manner to learn diverse and robust gait representations without expensive manual human annotations. Prompted by the increased use of the transformer model in all areas of deep learning, including computer vision, in this work we explore the use of five different vision transformer architectures directly applied to self-supervised gait recognition. We pretrain the simple ViT, CaiT, CrossFormer, Token2Token and TwinsSVT on two different large-scale gait datasets: GREW and DenseGait. We provide extensive results for zero-shot and fine-tuning on two benchmark gait recognition datasets, CASIA-B and FVG, and explore the relationship between the amount of spatial and temporal gait information used by the visual transformer. Our results show that in designing transformer models for processing motion, using a hierarchical approach (i.e. CrossFormer models) on finer grained movement fairs comparatively better than previous whole-skeleton approaches.*


### Introduction

In this work, we trained 5 different transformer models adapted for processing skeleton sequences. The network definitions is based on [vit-pytorch](https://github.com/lucidrains/vit-pytorch) repo, with some modifications. The models can be found in `models/` folder.

![](images/AllArchitectures.svg)


To adapt the various models to work with skeleton sequences, we upscale the skeletons using various methods. We chose to use [TSSI](https://arxiv.org/pdf/1909.05704.pdf) format and bicubic interpolation.

![](images/upsample-example-white.svg)

![](images/ViTPreprocessing.svg)


This work relies on [DenseGait](https://www.mdpi.com/1424-8220/22/18/6803), and [GREW](https://www.grew-benchmark.org/) for self-supervised training of models. Contact the authors for dataset access.

This repo is based on [acumen-template](https://github.com/cosmaadrian/acumen-template) to organise the project, and uses [wandb.ai](https://wandb.ai/) for experiment tracking.


### Experiments

All experimens in our paper can be run from the `experiments/` folder.

```
cd experiments/
bash train-evaluate-all.sh
```

### Citation

If you find our work helpful, you can cite it using the following:

```
TBD - Under Review
```



This work is based on our other works in self-supervised gait representation learning. If you find them helpful, please cite them:

[Learning Gait Representations with Noisy Multi-Task Learning](https://www.mdpi.com/1424-8220/22/18/6803)

```
@Article{cosma22gaitformer,
  AUTHOR = {Cosma, Adrian and Radoi, Emilian},
  TITLE = {Learning Gait Representations with Noisy Multi-Task Learning},
  JOURNAL = {Sensors},
  VOLUME = {22},
  YEAR = {2022},
  NUMBER = {18},
  ARTICLE-NUMBER = {6803},
  URL = {https://www.mdpi.com/1424-8220/22/18/6803},
  ISSN = {1424-8220},
  DOI = {10.3390/s22186803}
}
```

This work relies on our previous paper [WildGait: Learning Gait Representations from Raw Surveillance Streams](https://www.mdpi.com/1424-8220/21/24/8387). Please consider citing with:

```
@Article{cosma20wildgait,
  AUTHOR = {Cosma, Adrian and Radoi, Ion Emilian},
  TITLE = {WildGait: Learning Gait Representations from Raw Surveillance Streams},
  JOURNAL = {Sensors},
  VOLUME = {21},
  YEAR = {2021},
  NUMBER = {24},
  ARTICLE-NUMBER = {8387},
  URL = {https://www.mdpi.com/1424-8220/21/24/8387},
  PubMedID = {34960479},
  ISSN = {1424-8220},
  DOI = {10.3390/s21248387}
}
```
### License
This work is protected by CC BY-NC-ND 4.0 License (Non-Commercial & No Derivatives).
