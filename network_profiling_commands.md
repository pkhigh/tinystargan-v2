Please follow the instructions to install the required libraries in a new conda environment. Then download the datasets and all the pre-trained models. Activate the environment in the terminal before executing any commands.

For obtaining MACs and Size, follow the commands below:

StarGAN-v2 celeba_hq
```bash
python main.py --mode profile --num_domains 2 --w_hpf 1 
```

StarGAN-v2 afhq
```bash
python main.py --mode profile --num_domains 3 --w_hpf 0 
```

TinyStarGAN-v2 celeba_hq
```bash
python main.py --mode profile --num_domains 2 --w_hpf 1 \
               --alpha 128 --efficient 1 
```

TinyStarGAN-v2 afhq
```bash
python main.py --mode profile --num_domains 3 --w_hpf 0 \
               --alpha 128 --efficient 1
```