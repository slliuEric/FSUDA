# A Structure-aware Framework of Unsupervised Cross-Modality Domain Adaptation via Frequency and Spatial Knowledge Distillation
This is the official implementation of the paper"A Structure-aware Framework of Unsupervised Cross-Modality Domain Adaptation via Frequency and Spatial Knowledge Distillation“

A preliminary version of this work has been published as a conference paper [AAAI23' Reducing Domain Gap in Frequency and Spatial domain for Cross-modality Domain Adaptation on Medical Image Segmentation](https://arxiv.org/pdf/2211.15235.pdf)

# Abstract 
Unsupervised domain adaptation (UDA) aims
to train a model on a labeled source domain and adapt
it to an unlabeled target domain. In medical image segmentation field, most existing UDA methods rely on adversarial learning to address the domain gap between different image modalities. However, this process is complicated and inefficient. In this paper, we propose a simple
yet effective UDA method based on both frequency and
spatial domain transfer under a multi-teacher distillation
framework. In the frequency domain, we introduce nonsubsampled contourlet transform for identifying domaininvariant and domain-variant frequency components (DIFs
and DVFs) and replace the DVFs of the source domain
images with those of the target domain images while
keeping the DIFs unchanged to narrow the domain gap.
In the spatial domain, we propose a batch momentum
update-based histogram matching strategy to minimize the
domain-variant image style bias. Additionally, we further
propose a dual contrastive learning module at both image and pixel levels to learn structure-related information.
Our proposed method outperforms state-of-the-art methods on two cross-modality medical image segmentation
datasets (cardiac and abdominal). 

![image](https://github.com/slliuEric/FSUDA/assets/57536012/3889dd33-af73-49ab-bec5-106e39f7525a)


# Citation
If you use this code for your research, please cite our paper.

@article{liu2022reducing,,<br>
  title={Reducing Domain Gap in Frequency and Spatial domain for Cross-modality Domain Adaptation on Medical Image Segmentation},<br>
  author={Liu, Shaolei and Yin, Siqi and Qu, Linhao and Wang, Manning},<br>
  journal={arXiv preprint arXiv:2211.15235},<br>
  year={2022}<br>
}
