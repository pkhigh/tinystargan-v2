Please follow the instructions to install the required libraries in a new conda environment. Then download the datasets and all the pre-trained models. Activate the environment in the terminal before executing any commands.

For evaluating pre-trained models, follow the commands below:

StarGAN-v2 celeba_hq
```bash
python main.py --mode eval --num_domains 2 --w_hpf 1 \
               --resume_iter 100000 \
               --train_img_dir data/celeba_hq/train \
               --val_img_dir data/celeba_hq/val \
               --checkpoint_dir expr/checkpoints/celeba_hq \
               --eval_dir expr/eval/celeba_hq
```

StarGAN-v2 afhq
```bash
python main.py --mode eval --num_domains 3 --w_hpf 0 \
               --resume_iter 100000 \
               --train_img_dir data/afhq/train \
               --val_img_dir data/afhq/val \
               --checkpoint_dir expr/checkpoints/afhq \
               --eval_dir expr/eval/afhq
```

TinyStarGAN-v2 celeba_hq
```bash
python main.py --mode eval --num_domains 2 --w_hpf 1 \
               --resume_iter 100000 \
               --alpha 128 --efficient 1 \
               --train_img_dir data/celeba_hq/train \
               --val_img_dir data/celeba_hq/val \
               --checkpoint_dir expr/checkpoints/tiny_celeba_hq \
               --eval_dir expr/eval/tiny_celeba_hq
```

TinyStarGAN-v2 afhq
```bash
python main.py --mode eval --num_domains 3 --w_hpf 0 \
               --resume_iter 100000 \
               --alpha 128 --efficient 1 \
               --train_img_dir data/afhq/train \
               --val_img_dir data/afhq/val \
               --checkpoint_dir expr/checkpoints/tiny_afhq \
               --eval_dir expr/eval/tiny_afhq
```