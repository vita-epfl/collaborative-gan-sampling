# baseline
# python dynamic.py --nlayers=6 --teacher_name='default' --d_step=1 --dataset="Imbal-8Gaussians" --niters=5001 --mode='training' --ratio=0.8 --batch_size=400 --save_model && echo 'success'
# refinement
python dynamic.py --nlayers=6 --teacher_name='rollout' --rollout_method='ladam' --lrd=8e-3 --rollout_rate=5e-1 --rollout_steps=50 --d_step=1 --batch_size=400 --dataset="Imbal-8Gaussians" --ckpt_num=2000 --niters=10001 --mode='refinement' --refine_method='optimized_fake' && echo 'success'
#
#
#
# python dynamic.py --nlayers=6 --teacher_name='default' --d_step=1 --dataset="Imbal-8Gaussians" --niters=5001 --mode='training' --ratio=0.8 --batch_size=100 --save_model && echo 'success'