# 2D Synthetic Data

### Scripts

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
python compare.py --mode="benchmark" --dataset="Imbal-8Gaussians"
```

Discriminator Diagnosis
```
python compare.py --mode="calibrate" --dataset="Imbal-8Gaussians"
python compare.py --mode="diagnostic" --dataset="Imbal-8Gaussians"
```