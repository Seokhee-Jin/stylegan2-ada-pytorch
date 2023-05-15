#### conda env #####
conda deactivate
conda remove -n stylegan2_ada --all
conda create -n stylegan2_ada python=3.7 
conda activate stylegan2_ada

#### requirements #####
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3
pip(conda?) install psutil
conda install scipy

* When importError~~ occurred, try this:
https://velog.io/@ssw9999/ImportError-libx8664-linux-gnulibstdc.so.6-version-GLIBCXX3.4.29-not-found


0. Create dataset

python dataset_tool.py --source /home/hail/PycharmProjects/seokhee_jin/data/dataset/final_5000 --dest /home/hail/PycharmProjects/seokhee_jin/data/dataset/final_5000_dataset.zip


1. train
python train.py --outdir /home/hail/PycharmProjects/seokhee_jin/data/output --data /home/hail/PycharmProjects/seokhee_jin/data/dataset/final_5000_dataset.zip


--resume=ffhq256 --snap=10 --gamma=10 
*gamma: 0.16 ~ 0.82 ~ 4.1


#### docker #####

#building dockerfile
cd dir
docker login -u jin749
docker build -t jin749/stylegan2_ada:1.0 .

# run and 
* it: interacive terminal
sudo docker run -it stylegan2_ada:1.0 sh
nvidia-docker run --runtime=nvidia --gpus all -it stylegan2_ada:1.0 sh

# exit
[ctrl + d]

sudo docker run --gpus all -it -rm -v `pwd`:/tmp -w /tmp datafireball/stylegan2:v0.1

cd stylegan2-ada-pytorch/

python train.py --outdir ./final_5000_out --data ./final_5000.zip


sudo nvidia-docker run -it --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 -u $(id -u):$(id -g) -v /home/hail/:/mnt --device /dev/nvidia0:/dev/nvidia0 heatonresearch/stylegan2-ada /bin/bash

sudo nvidia-docker run -it --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 -u $(id -u):$(id -g) -v /home/hail/:/mnt --device /dev/nvidia0:/dev/nvidia0 jin749/stylegan2_ada:1.0 /bin/bash


