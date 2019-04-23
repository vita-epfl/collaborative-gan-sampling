ROLLOUT_STEPS=50
ROLLOUT_RATE=0.1
python main.py --mode="refinement" --rollout_rate=${ROLLOUT_RATE} --rollout_steps=${ROLLOUT_STEPS} --rollout_method="ladam" --lrd=8e-3 --dataset="Imbal-8Gaussians" --ckpt_num=3000 --niters=5001 --shaping_method="optimized_fake"