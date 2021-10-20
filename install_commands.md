Create a new conda environment and follow the following commands:

```bash
conda create -n tinystargan-v2 python=3.6.7
conda activate tinystargan-v2
conda install -y pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.2.89 -c pytorch
conda install x264=='1!152.20180806' ffmpeg=4.3.1 -c conda-forge
pip install opencv-python==4.4.0.44 ffmpeg-python==0.2.0 scikit-image==0.17.2
pip install pillow==8.0.0 scipy==1.5.2 tqdm==4.50.2 munch==2.5.0
pip install torchprofile
```