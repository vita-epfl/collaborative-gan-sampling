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

## Usage:
Download dataset with:

    $ python download.py mnist celebA

## 2d synthetic:
- cd teacher
- sh run.sh
- sh run2.sh
- sh run_imbal.sh

## MNIST:
- cd dcgan/DCGAN-tensorflow-master
- To Train 
    $ python ns_main.py 
 
- To Collaboratively Sample
    $ python ns_main.py --mode "refinement" --teacher_name "gpurollout" --epoch 2

## CelebA:
