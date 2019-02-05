# Main Flags

We enumerate some of the most important flags of `ns_main.py` and `dc_main.py`:

- `--mode 'training'`: If training, trains the GAN. If refinement, shapes the  discriminator. If testing, collaboratively samples  
- `--denoise False`: Sets the application to Image Denoising
- `--use_refined True`: Uses the refined samples to shape the discriminator. If False, uses the default generated samples  
- `--epoch 5`: Total number of epochs to run the `mode`
- `--load_epoch 400`: If not training, the iterations/epoch to load the saved model from
- `--load_model_dir path_to_saved_model\`: The directory of saved model
- `--refine_D_iters 1`: Used in refinement mode, to switch from 'refinement' to 'testing' upon complete of `refine_D_iters` of shaping D
- `--rollout_steps 100`: The number of rollout steps (k)
- `--rollout_rate 50`: The step_size of each rollout step  
- `--rollout_method momentum`: The optimization algorithm to roll out the samples

