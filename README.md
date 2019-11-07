# Collaborative Sampling in Generative Adversarial Networks

This repository contains the code for the paper: 

[Collaborative Sampling in Generative Adversarial Networks](https://arxiv.org/abs/1902.00813)

<img src="assets/diagram.png">

Once GAN training completes, we use both the generator and the discriminator to produce samples **collaboratively**. Our sampling scheme consists of one sample proposal step and multiple sample refinement steps. (I) The fixed generator proposes samples. (II) Subsequently, the discriminator provides gradients, with respect to activation maps of the proposed samples, back to a particular layer of the generator. Gradient-based updates of the activation maps are performed repeatedly until the samples are classified as *real* by the discriminator.

<br>

## [2D Synthetic](2D/README.md)

Different sampling methods on a synthetic *imbalanced* mixture of 8 Gaussians. Standard GAN training is prone to mode collapse. Our collaborative sampling scheme applied to early terminated GANs succeeds in recovering all modes without compromising sample quality, outperforming the rejection sampling and metropolis-hastings method. 

| Real | [GAN](https://papers.nips.cc/paper/5423-generative-adversarial-nets) <br> 1K Iter | [GAN](https://papers.nips.cc/paper/5423-generative-adversarial-nets) <br> 9K Iter | [DRS](https://arxiv.org/abs/1810.06758) <br> at 1K Iter | [MH-GAN](https://arxiv.org/abs/1810.06758) <br> at 1K Iter | Refine <br> at 1K Iter | Collab <br> at 1K Iter |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|![](assets/2d/sample_real.png) | ![](assets/2d/sample_early.png) | ![](assets/2d/sample_collapse.png) | ![](assets/2d/sample_reject.png) | ![](assets/2d/sample_mh.png) | ![](assets/2d/sample_refine.png) | ![](assets/2d/sample_collab.png) 
![](assets/2d/kde_real.png) | ![](assets/2d/kde_early.png) | ![](assets/2d/kde_collapse.png) | ![](assets/2d/kde_reject.png) | ![](assets/2d/kde_mh.png) | ![](assets/2d/kde_refine.png) | ![](assets/2d/kde_collab.png) |


| Quality | Diversity | Overall |
| ------------- |:-------------:|:-------------:|
![](assets/2d/Imbal-8Gaussians_benchmark_2d_good.png) | ![](assets/2d/Imbal-8Gaussians_benchmark_2d_kl.png) | ![](assets/2d/Imbal-8Gaussians_benchmark_2d_js.png) |

## [Nature Images](image/README.md)
![](assets/celebA/1.png) ![](assets/celebA/2.png) ![](assets/celebA/3.png) ![](assets/celebA/4.png) ![](assets/celebA/5.png) ![](assets/celebA/6.png) ![](assets/celebA/7.png) ![](assets/celebA/8.png) ![](assets/celebA/9.png) 

DCGAN on the CelebA. (Top) Samples from standard sampling. (Middle) Samples from our collaboratively sampling method. (Bottom) The difference between the top and middle row images, reflecting the effectiveness of our method in improving the quality of natural images.

| Dataset | IS | FID | 
|:-------------:|:-------------:|:-------------:|
| Cifar10 |![](assets/cifar/benchmark_is.png) | ![](assets/cifar/benchmark_fid.png) |
| CelebA |![](assets/celebA/benchmark_is.png) | ![](assets/celebA/benchmark_fid.png) |

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

## Citation
If you find the codes or paper useful for your research, please cite our paper:
```
@techreport{liu2019collaborative,
  title={Collaborative Sampling in Generative Adversarial Networks},
  author={Liu, Yuejiang and Kothari, Parth Ashit and Alahi, Alexandre},
  year={2019}
}
```

<br>

## Acknowledgements
The baseline implementation has been based on [this repository](https://github.com/carpedm20/DCGAN-tensorflow)
