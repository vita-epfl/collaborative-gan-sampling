# Collaborative Sampling in GANs for Natural Images

## Download  
```
python download.py mnist celebA
mkdir model 
wget https://github.com/tensorflow/models/raw/master/research/gan/mnist/data/classify_mnist_graph_def.pb -P model/
```

## NSGAN
```
python ns_main.py --mode "training" --epoch 20 -save_model True --imbalance True
python ns_main.py --mode "refinement" --refiner_name "gpurollout" --imbalance True --epoch 3  --load_model_dir checkpoints/mnist/epoch_20_refiner_default_rollout_method_momentum_rollout_steps_50_rollout_rate_50.00000/ --load_epoch 5 --save_figs --rollout_steps=50 --rollout_rate=0.5 --save_model --collab_layer 2 
python ns_main.py --mode "testing" --refiner_name "gpurollout" --imbalance True --epoch 0 --load_model_dir checkpoints/mnist/epoch_3_refiner_gpurollout_rollout_method_momentum_rollout_steps_50_rollout_rate_0.50000/ --load_epoch 2 --save_figs --rollout_steps=50 --rollout_rate=0.5 --collab_layer 2 
```

## DCGAN
```
python dc_main.py --mode "training" --epoch 50 --G_it 2 --crop --save_model
python dc_main.py --mode "refinement" --crop --refiner_name "gpurollout" --epoch 2 --load_model_dir dc_checkpoints/celebA/epoch_50_refiner_default_rollout_method_momentum_rollout_steps_10_rollout_rate_5.00000/celebA_100_64_64/ --load_epoch 30 --rollout_steps=20 --rollout_rate=0.1 --collab_layer=1 
python dc_main.py --dataset celebA --crop --mode "testing" --teacher_name "gpurollout" --epoch 0 --load_model_dir dc_checkpoints/celebA/epoch_2_refiner_gpurollout_rollout_method_momentum_rollout_steps_20_rollout_rate_0.10000/celebA_100_64_64/ --load_epoch 1 --rollout_steps=20 --rollout_rate=0.1 --collab_layer=1 --save_figs 
```

## WGAN-GP
```
python wgan_main.py --mode "training" --epoch 20 -save_model True --imbalance True 
python wgan_main.py --mode "refinement" --refiner_name "gpurollout" --imbalance True --epoch 1  --load_model_dir checkpoints/mnist/epoch_20_refiner_default_rollout_method_momentum_rollout_steps_10_rollout_rate_0.10000/ --load_epoch 15 --save_figs --rollout_steps=50 --rollout_rate=0.1 --save_model --collab_layer 2 
python wgan_main.py --mode "testing" --refiner_name "gpurollout" --imbalance True --epoch 0 --load_model_dir checkpoints/mnist/epoch_1_refiner_gpurollout_rollout_method_momentum_rollout_steps_50_rollout_rate_0.10000/ --load_epoch 0 --save_figs --rollout_steps=50 --rollout_rate=0.1 --collab_layer 2 
```
