# VLM-Transformers
This repository includes the creation of vision language model from scratch, applying deep learning and Pytorch library techniques

#First of all we are going to install the Pytorch and check if we have CUDA enabled gpu.

* Install the Pytorch with CUDA support:
    $ conda activate torch_env
    $ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia 
    $ conda install ipykernel

#Second we are going to check if we have CUDA enabled gpu.
    Run the check.py file:
    $ python check.py

CUDA is available
CUDA version: 12.1
CUDA capabilities: (8, 6)
CUDA devices: 1
CUDA current device: 0
CUDA device name: NVIDIA GeForce RTX 3050 6GB Laptop GPU
CUDA max memory: 6.441926656 GB

If Cuda is not avalable then we can not create the vision language model from scratch
becuase it will take to much time and memory to train the model on CPU

