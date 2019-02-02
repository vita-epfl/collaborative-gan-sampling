default:
	python main.py --D_LR 1 --G_LR 1 --D_it 1 --G_it 1
	python main.py --D_LR 1 --G_LR 2 --D_it 1 --G_it 1
	python main.py --D_LR 1 --G_LR 3 --D_it 1 --G_it 1
	python main.py --D_LR 3 --G_LR 1 --D_it 1 --G_it 1
	python main.py --D_LR 2 --G_LR 1 --D_it 1 --G_it 1

	python main.py --D_LR 1 --G_LR 1 --D_it 1 --G_it 2
	python main.py --D_LR 1 --G_LR 0.5 --D_it 1 --G_it 2
	python main.py --D_LR 1 --G_LR 0.3 --D_it 1 --G_it 2
	python main.py --D_LR 2 --G_LR 1 --D_it 1 --G_it 2
	python main.py --D_LR 3 --G_LR 1 --D_it 1 --G_it 2

	python main.py --D_LR 1 --G_LR 1 --D_it 2 --G_it 1
	python main.py --D_LR 1 --G_LR 3 --D_it 2 --G_it 1
	python main.py --D_LR 1 --G_LR 5 --D_it 2 --G_it 1
	python main.py --D_LR 1 --G_LR 0.5 --D_it 2 --G_it 1
	python main.py --D_LR 1 --G_LR 0.3 --D_it 2 --G_it 1
	python main.py --D_LR 2 --G_LR 1 --D_it 2 --G_it 1
	@echo "All Done."

paper:
	# python ns_main.py --rollout_steps 1 --rollout_rate 1  
	python ns_main.py --rollout_steps 100 --rollout_rate 1  
	python ns_main.py --rollout_steps 100 --rollout_rate 5
	python ns_main.py --rollout_steps 100 --rollout_rate 10
	python ns_main.py --rollout_steps 100 --rollout_rate 50
	python ns_main.py --rollout_steps 100 --rollout_rate 100
	python ns_main.py --rollout_steps 100 --rollout_rate 200
	python ns_main.py --rollout_steps 50 --rollout_rate 1  
	python ns_main.py --rollout_steps 50 --rollout_rate 5
	python ns_main.py --rollout_steps 50 --rollout_rate 10
	python ns_main.py --rollout_steps 50 --rollout_rate 50
	python ns_main.py --rollout_steps 50 --rollout_rate 100
	python ns_main.py --rollout_steps 50 --rollout_rate 200