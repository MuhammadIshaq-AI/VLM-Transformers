#This script is used to check if the CUDA is available
import torch
import sys
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device being used: {device}")

if device == 'cuda':
    print('CUDA is available')
    print('CUDA version:', torch.version.cuda)
    print('CUDA capabilities:', torch.cuda.get_device_capability())
    print('CUDA devices:', torch.cuda.device_count())
    print('CUDA current device:', torch.cuda.current_device())
    print('CUDA device name:', torch.cuda.get_device_name())
    props = torch.cuda.get_device_properties(0)
    print(f'CUDA max memory: {props.total_memory / 1e9:.2f} GB')
else:
    print('CUDA is not available, using CPU.')
    print('Python version:', sys.version)
    print('Torch version:', torch.__version__)