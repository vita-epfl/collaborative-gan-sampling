# Collaborative Sampling in Generative Adversarial Networks

> The standard practice in using Generative Adversarial Networks (GANs) discards the discriminator. However, this sampling method loses valuable information learned by the discriminator regarding the data distribution. In this work, we propose a collaborative sampling scheme between the generator and the discriminator for improved data generation. Guided by the discriminator, our approach refines the generated samples through gradient-based updates at a particular layer of the generator, shifting the generator distribution closer to the real data distribution. Additionally, we introduce a practical discriminator shaping method that can smoothen the loss landscape provided by the discriminator for effective sample refinement. Through extensive experiments on synthetic and image datasets, we demonstrate that our proposed method can improve generated samples both quantitatively and qualitatively, offering a new degree of freedom in GAN sampling. 

```
@InProceedings{liu2019collaborative,
  title={Collaborative Sampling in Generative Adversarial Networks},
  author={Liu, Yuejiang and Kothari, Parth Ashit and Alahi, Alexandre},
  booktitle = {Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI)},
  month = {February},
  year = {2020}
}
```

<br>

## Overview

<img src="assets/diagram.png">

Once GAN training completes, we use both the generator and the discriminator to produce samples *collaboratively*. Our sampling scheme consists of one sample proposal step and multiple sample refinement steps. (I) The fixed generator proposes samples. (II) Subsequently, the discriminator provides gradients, with respect to activation maps of the proposed samples, back to a particular layer of the generator. Gradient-based updates of the activation maps are performed repeatedly until the samples are classified as *real* by the discriminator.

<br>

## [2D Synthetic](2D/README.md)

GANs for modelling an *imbalanced* mixture of 8 Gaussians. Vanilla [GANs](https://papers.nips.cc/paper/5423-generative-adversarial-nets) are prone to mode collapse. The accept-reject sampling algorithms including Discriminator Rejection Sampling ([DRS](https://arxiv.org/abs/1810.06758)) and Metropolis-Hastings method ([MH-GAN](https://arxiv.org/abs/1810.06758)) suffer from severe distribution bias due to the mismatch between distribution supports. Our collaborative sampling scheme applied to early terminated GANs succeeds in recovering all modes without compromising sample quality, significantly outperforming the baseline methods.

| Real | GAN <br> 1K Iter | GAN <br> 9K Iter | DRS <br> at 1K Iter | MH-GAN <br> at 1K Iter | Refine <br> at 1K Iter | Collab <br> at 1K Iter |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|![](assets/2d/sample_real.png) | ![](assets/2d/sample_early.png) | ![](assets/2d/sample_collapse.png) | ![](assets/2d/sample_reject.png) | ![](assets/2d/sample_mh.png) | ![](assets/2d/sample_refine.png) | ![](assets/2d/sample_collab.png) 
![](assets/2d/kde_real.png) | ![](assets/2d/kde_early.png) | ![](assets/2d/kde_collapse.png) | ![](assets/2d/kde_reject.png) | ![](assets/2d/kde_mh.png) | ![](assets/2d/kde_refine.png) | ![](assets/2d/kde_collab.png) |

| Quality | Diversity | Overall |
| ------------- |:-------------:|:-------------:|
![](assets/2d/Imbal-8Gaussians_benchmark_2d_good.png) | ![](assets/2d/Imbal-8Gaussians_benchmark_2d_kl.png) | ![](assets/2d/Imbal-8Gaussians_benchmark_2d_js.png) |

<br>

## [Image Generation](image/README.md)
![](assets/celebA/1.png)![](assets/celebA/2.png)![](assets/celebA/3.png)![](assets/celebA/4.png)![](assets/celebA/5.png)![](assets/celebA/7.png)![](assets/celebA/8.png)![](assets/celebA/9.png)![](assets/celebA/10.png)![](assets/celebA/12.png)![](assets/celebA/13.png)![](assets/celebA/14.png)![](assets/celebA/15.png)![](assets/celebA/16.png)

DCGAN for modelling human faces on the CelebA dataset. (Top) Samples from standard sampling. (Middle) Samples from our collaborative sampling method. (Bottom) The difference between the top and the middle row.

| Dataset | IS | FID | 
|:-------------:|:-------------:|:-------------:|
| Cifar10 |![](assets/cifar/benchmark_is.png) | ![](assets/cifar/benchmark_fid.png) |
| CelebA |![](assets/celebA/benchmark_is.png) | ![](assets/celebA/benchmark_fid.png) |

<br>

## [Image Manipulation](image/README.md)
![](assets/cycle/cycle_1.jpg)![](assets/cycle/cycle_2.jpg)![](assets/cycle/cycle_3.jpg)

CycleGAN for unpaired image-to-image translation. (Top) Samples from standard sampling. (Middle) Samples from our collaborative sampling method. (Bottom) The difference between the top and the middle row.

<br>

## Dependencies:
 
- tensorflow==1.13.0
- CUDA==10.0
- pillow
- scipy=1.2
- matplotlib
- requests
- tqdm 

<br>

## Acknowledgements
The baseline implementation has been based on [this repository](https://github.com/carpedm20/DCGAN-tensorflow)
