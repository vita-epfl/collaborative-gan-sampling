# Prerequisites:

scipy
tensorflow v1.9.0
CuDNN=7.0.5 
matplotlib
requests
tqdm 

# Usage:
For downloading data
python download.py mnist celebA
wget https://github.com/tensorflow/models/blob/master/research/gan/mnist/data/classify_mnist_graph_def.pb

# 2d synthetic:
cd teacher
sh run.sh
sh run2.sh
sh run_imbal.sh

# MNIST:
cd dcgan/DCGAN-tensorflow-master
sh run_ns_gan.sh
