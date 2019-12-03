# NSGAN on Mnist

### Scripts

Download
```
python download.py "mnist"
wget https://github.com/tensorflow/models/raw/master/research/gan/mnist/data/classify_mnist_graph_def.pb -P external/
```

Train
```
sh run_train.sh
```

Rejection Sampling & Metropolis-Hastings
```
sh run_calibrate.sh
```

Collaborative Sampling
```
sh run_shaping.sh
```

## Evaluation

Sampling Method Benchmark
```
python compare.py --mode="benchmark" --logdir="logs/"
```

Discriminator Shaping
```
python compare.py --mode="shape" --logdir="logs/"
```