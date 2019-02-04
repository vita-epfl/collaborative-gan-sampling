## Prerequisites:
 
- tensorflow==1.9.0
- CuDNN=9.0 
- scipy
- matplotlib
- requests
- tqdm 
- [MNIST IS Model](https://github.com/tensorflow/models/blob/master/research/gan/mnist/data/classify_mnist_graph_def.pb)

## Usage:
Download dataset with:

    $ python download.py mnist celebA


# 2d synthetic:
cd teacher
sh run.sh
sh run2.sh
sh run_imbal.sh

# MNIST:
cd dcgan/DCGAN-tensorflow-master
sh run_ns_gan.sh
