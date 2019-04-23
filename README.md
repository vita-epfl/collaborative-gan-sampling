# Collaborative Sampling in Generative Adversarial Networks

This repository contains the code for the paper: 

[Collaborative Sampling in Generative Adversarial Networks](https://arxiv.org/abs/1902.00813)

<img src="assets/diagram.png">

Once GAN training completes, we use both the generator and the discriminator to produce samples **collaboratively**. Our sampling scheme consists of one sample proposal step and multiple sample refinement steps. (I) The fixed generator proposes samples. (II) Subsequently, the discriminator provides gradients, with respect to activation maps of the proposed samples, back to a particular layer of the generator. Gradient-based updates of the activation maps are performed repeatedly until the samples are classified as *real* by the discriminator.

<br>

## [2D Synthetic](2D/README.md)

![](assets/2d/3k.png) | ![](assets/2d/20k.png) | ![](assets/2d/rejection.png) | ![](assets/2d/collaborative.png)
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
3K iterations <br /> [standard sampling](https://papers.nips.cc/paper/5423-generative-adversarial-nets) | 20K iterations <br /> [standard sampling](https://papers.nips.cc/paper/5423-generative-adversarial-nets) | 3K iterations <br /> [rejection sampling](https://arxiv.org/abs/1810.06758) | 3K iterations <br /> our [collaborative sampling](https://arxiv.org/abs/1902.00813)

NSGAN on a synthetic *imbalanced* mixture of 8 Gaussians. Standard GAN training is prone to mode collapse. Our collaborative sampling scheme applied to early terminated GANs succeeds in recovering all modes without compromising sample quality, significantly outperforming the rejection sampling method. 

<br>

## [Nature Images](image/README.md)
![](assets/celebA/1.png) ![](assets/celebA/2.png) ![](assets/celebA/3.png) ![](assets/celebA/4.png) ![](assets/celebA/5.png) ![](assets/celebA/6.png) ![](assets/celebA/7.png) ![](assets/celebA/8.png) ![](assets/celebA/9.png) 

DCGAN on the CelebA. (Top) Samples from standard sampling. (Middle) Samples from our collaboratively sampling method. (Bottom) The difference between the top and middle row images, reflecting the effectiveness of our method in improving the quality of natural images.

<br>

## Dependencies:
 
- tensorflow==1.13.0
- CUDA==10.0
- pillow
- scipy
- matplotlib
- requests
- tqdm 

<br>

## Citation
If you find the codes or paper useful for your research, please cite our paper:
```
@misc{liu2019collaborative,
  title={Collaborative Sampling in Generative Adversarial Networks},
  author={Liu, Yuejiang and Kothari, Parth Ashit and Alahi, Alexandre},
  year={2019},
  Eprint = {arXiv:1902.00813}
}
```

<br>

## Acknowledgements
The baseline implementation has been based on [this repository](https://github.com/carpedm20/DCGAN-tensorflow)

