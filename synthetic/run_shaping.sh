rm -rf log/Imbal-8Gaussians_shape/
python main.py --dataset="Imbal-8Gaussians" --ratio=0.9 --niters=20001 --mode="shape" --method="benchmark" --eval_type="logs" --ckpt_num=1000 --lrd=8e-3 --eval_every=1000
python main.py --dataset="Imbal-8Gaussians" --ratio=0.9 --niters=20001 --mode="shape" --method="benchmark" --eval_type="logs" --ckpt_num=3000 --lrd=8e-3 --eval_every=1000
python main.py --dataset="Imbal-8Gaussians" --ratio=0.9 --niters=20001 --mode="shape" --method="benchmark" --eval_type="logs" --ckpt_num=5000 --lrd=8e-3 --eval_every=1000
python main.py --dataset="Imbal-8Gaussians" --ratio=0.9 --niters=20001 --mode="shape" --method="benchmark" --eval_type="logs" --ckpt_num=7000 --lrd=8e-3 --eval_every=1000
python main.py --dataset="Imbal-8Gaussians" --ratio=0.9 --niters=20001 --mode="shape" --method="benchmark" --eval_type="logs" --ckpt_num=9000 --lrd=8e-3 --eval_every=1000
# #
# rm -rf log/25Gaussians_shape/
# python main.py --dataset="25Gaussians" --niters=20001 --mode="shape" --method="benchmark" --eval_type="logs" --ckpt_num=100000 --lrd=10e-3 --rollout_rate=0.05 --nhidden=256 --nlayers=6 --scale=1.0 --eval_every=1000
# python main.py --dataset="25Gaussians" --niters=20001 --mode="shape" --method="benchmark" --eval_type="logs" --ckpt_num=300000 --lrd=10e-3 --rollout_rate=0.05 --nhidden=256 --nlayers=6 --scale=1.0 --eval_every=1000
# python main.py --dataset="25Gaussians" --niters=20001 --mode="shape" --method="benchmark" --eval_type="logs" --ckpt_num=500000 --lrd=10e-3 --rollout_rate=0.05 --nhidden=256 --nlayers=6 --scale=1.0 --eval_every=1000
# python main.py --dataset="25Gaussians" --niters=20001 --mode="shape" --method="benchmark" --eval_type="logs" --ckpt_num=700000 --lrd=10e-3 --rollout_rate=0.05 --nhidden=256 --nlayers=6 --scale=1.0 --eval_every=1000
# python main.py --dataset="25Gaussians" --niters=20001 --mode="shape" --method="benchmark" --eval_type="logs" --ckpt_num=900000 --lrd=10e-3 --rollout_rate=0.05 --nhidden=256 --nlayers=6 --scale=1.0 --eval_every=1000