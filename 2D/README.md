# Collaborative Sampling in GANs for 2D Synthetic Data

## Training
```
python main.py --dataset=Imbal-8Gaussians --niters=5001 --mode="training" --save_model
```

## Collaborative Sampling
```
python main.py --mode="refinement" --rollout_rate=0.1 --rollout_steps=50 --rollout_method="ladam" --lrd=8e-3 --dataset="Imbal-8Gaussians" --ckpt_num=3000 --niters=5001 --shaping_method="optimized_fake"
```

## Rejection Sampling
```
python main.py --mode="rejection" --niters=1 --dataset="Imbal-8Gaussians" --ckpt_num=3000 
```

