Please follow the instructions to install the required libraries in a new conda environment. Then download the datasets and all the pre-trained models. Activate the environment in the terminal before executing any commands.

NOTE: Each training experiment with default hyper-parameters requires a Tesla V100 32 GB GPU and runs for 3 days.

For training models, follow the commands below:

StarGAN-v2 on celeba_hq
```bash
python main.py --mode train --num_domains 2 --w_hpf 1 \
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 \
               --train_img_dir data/celeba_hq/train \
               --val_img_dir data/celeba_hq/val
```

TinyStarGAN-v2 at alpha = 128 without distillation on celeba_hq
```bash
python main.py --mode train --num_domains 2 --w_hpf 1 \
               --alpha 128 --efficient 1 \
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 \
               --train_img_dir data/celeba_hq/train \
               --val_img_dir data/celeba_hq/val
```

TinyStarGAN-v2 at alpha = 128 with distillation on celeba_hq
```bash
python main.py --mode distill_train --num_domains 2 --w_hpf 1 \
               --alpha 128 --efficient 1 --f_lr 1e-4 \
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 \
               --teacher_checkpoint_dir expr/checkpoints/celeba_hq --teacher_resume_iter 100000 \
               --train_img_dir data/celeba_hq/train \
               --val_img_dir data/celeba_hq/val
```

StarGAN-v2 on afhq
```bash
python main.py --mode train --num_domains 3 --w_hpf 0 \
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 2 --lambda_cyc 1 \
               --train_img_dir data/afhq/train \
               --val_img_dir data/afhq/val
```

TinyStarGAN-v2 at alpha = 128 without distillation on afhq
```bash
python main.py --mode train --num_domains 3 --w_hpf 0 \
               --alpha 128 --efficient 1 \
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 2 --lambda_cyc 1 \
               --train_img_dir data/afhq/train \
               --val_img_dir data/afhq/val
```

TinyStarGAN-v2 at alpha = 128 with distillation on afhq
```bash
python main.py --mode distill_train --num_domains 3 --w_hpf 0 \
               --alpha 128 --efficient 1 --f_lr 1e-4 \
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 2 --lambda_cyc 1 \
               --teacher_checkpoint_dir expr/checkpoints/afhq --teacher_resume_iter 100000 \
               --train_img_dir data/afhq/train \
               --val_img_dir data/afhq/val
```