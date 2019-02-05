# Collaborative GAN Sampling

This repository contains the code for the paper: 

[Collaborative GAN Sampling](https://arxiv.org/abs/1902.00813)

Please cite this paper if you use the code in this repository as part of a published research project.

## Overview
<img src="https://github.com/vita-epfl/collaborative-gan-sampling/raw/master/images/IntroImage.png" height="400" width="600">

We introduce a collaborative sampling scheme between the generator and discriminator for improved sample generation. Once GAN training completes, we freeze the parameters of the generator and refine the generated samples leveraging on the discriminator gradients. We further propose to shape the discriminator loss landscape using these refined samples. Through sample-wise optimization, our method shifts the model distribution closer to the real data distribution.

## Prerequisites:
 
- tensorflow==1.9.0
- CUDA==9.0
- CuDNN=7.0.5 
- pillow
- scipy
- matplotlib
- requests
- tqdm 
- [MNIST IS Model](https://github.com/tensorflow/models/blob/master/research/gan/mnist/data/classify_mnist_graph_def.pb)

## Datasets:
Download datasets with:

    $ python download.py mnist celebA

# 2D Synthetic Datasets

## Commandline  

```
cd teacher
sh run.sh
sh run2.sh
sh run_imbal.sh
```

## Results 

### Schemes: 

a. *GAN*: Standard sampling from GAN only using the trained generator

b. *G2N-freeze*: Collaborative sampling from both the generator and discriminator, without additional discriminator training 

c. *G2N-tuning*: Collaborative sampling from both the generator and discriminator, with discriminator tuning using generated samples

d. *G2N-shaping*: Collaborative sampling from both the generator and discriminator, with discriminator shaping using refined samples

### 2D Gaussians 
<img src="https://github.com/vita-epfl/collaborative-gan-sampling/raw/master/images/2DGauss.png" height="400" width="600">

<br>

### SwissRoll
<img src="https://github.com/vita-epfl/collaborative-gan-sampling/raw/master/images/SwissRoll.png" height="400" width="600">

<br>

### Preventing Mode Collapse

<img src="https://github.com/vita-epfl/collaborative-gan-sampling/raw/master/images/ModeCollapse.png" height="400" width="400">

<br>

# MNIST
## Commandline 
- cd dcgan/DCGAN-tensorflow-master
- To Train 
```
    $ python ns_main.py 
```
 
- To Collaboratively Sample
```
   $ python ns_main.py --mode "refinement" --teacher_name "gpurollout" --epoch 2
```

## Results

### After 800 iterations 
<img src="https://github.com/vita-epfl/collaborative-gan-sampling/raw/master/images/it1200.png" height="200" width="600">

<br>

### After 1200 iterations 
<img src="https://github.com/vita-epfl/collaborative-gan-sampling/raw/master/images/it800.png" height="200" width="600">

<br>

### After 4000 iterations
<img src="https://github.com/vita-epfl/collaborative-gan-sampling/raw/master/images/it4000.png" height="200" width="600">

<br>

### Quantitative Comparison
<img src="https://github.com/vita-epfl/collaborative-gan-sampling/raw/master/images/QuantitativeComparison.png" height="300" width="600">

<br>

### Application in Denoising
<img src="https://github.com/vita-epfl/collaborative-gan-sampling/raw/master/images/Denoise.png" height="200" width="600">

<br>

# CelebA
## Commandline 
- cd dcgan/DCGAN-tensorflow-master
- To Train 
```
    $ python dc_main.py --G_it 2
```
 
- To Collaboratively Sample
```
   $ python dc_main.py --mode "refinement" --teacher_name "gpurollout" --epoch 2
```

##Results

<img src="https://github.com/vita-epfl/collaborative-gan-sampling/raw/master/images/CelebA.png" height="150" width="800">

<br>

## Acknowledgements
The baseline implementation has been based on [this repository](https://github.com/carpedm20/DCGAN-tensorflow)



