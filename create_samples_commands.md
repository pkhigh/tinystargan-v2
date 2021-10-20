Please follow the instructions to install the required libraries in a new conda environment. Then download the datasets and all the pre-trained models. Activate the environment in the terminal before executing any commands.

For creating samples from assets folder, follow the commands below:

StarGAN-v2 celeba_hq
```bash
python main.py --mode sample --num_domains 2 --w_hpf 1 \
               --resume_iter 100000 \
               --checkpoint_dir expr/checkpoints/celeba_hq \
               --val_batch_size 4 --latent_sample_per_domain 1 \
               --filename celeba_hq.jpg \
               --src_dir assets/paper/celeba_hq/src \
               --ref_dir assets/paper/celeba_hq/ref
```

StarGAN-v2 afhq
```bash
python main.py --mode sample --num_domains 3 --w_hpf 0 \
               --resume_iter 100000 \
               --checkpoint_dir expr/checkpoints/afhq \
               --val_batch_size 4 --latent_sample_per_domain 1 \
               --filename afhq.jpg \
               --src_dir assets/paper/afhq/src \
               --ref_dir assets/paper/afhq/ref
```

TinyStarGAN-v2 celeba_hq
```bash
python main.py --mode sample --num_domains 2 --w_hpf 1 \
               --alpha 128 --efficient 1 \
               --resume_iter 100000 \
               --checkpoint_dir expr/checkpoints/tiny_celeba_hq \
               --val_batch_size 4 --latent_sample_per_domain 1 \
               --filename tiny_celeba_hq.jpg \
               --src_dir assets/paper/celeba_hq/src \
               --ref_dir assets/paper/celeba_hq/ref
```

TinyStarGAN-v2 afhq
```bash
python main.py --mode sample --num_domains 3 --w_hpf 0 \
               --alpha 128 --efficient 1 \
               --resume_iter 100000 \
               --checkpoint_dir expr/checkpoints/tiny_afhq \
               --val_batch_size 4 --latent_sample_per_domain 1 \
               --filename tiny_afhq.jpg \
               --src_dir assets/paper/afhq/src \
               --ref_dir assets/paper/afhq/ref
```