# VLM-Transformers
This repository includes the creation of a Vision Language Model from scratch, applying deep learning and PyTorch library techniques.

## Project Structure
```text
VLM-Transformers/
├── src/                    # Core source code
│   ├── models/             # Model architectures (Vision Encoder, LLM, Projection)
│   ├── data/               # Data loading and preprocessing logic
│   ├── utils/              # Helper functions (logging, image processing)
│   └── training/           # Training loops and optimization logic
├── scripts/                # Utility scripts (GPU checks, evaluation)
├── configs/                # Configuration files (hyperparameters)
├── notebooks/              # Jupyter Notebooks for experimentation
├── tests/                  # Unit tests
├── .gitignore              # Git ignore file
├── requirements.txt        # Dependencies
└── README.md
```

## Setup Instructions

* **Install PyTorch with CUDA support:**
    ```bash
    conda activate torch_env
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia 
    conda install ipykernel
    ```

* **Verify CUDA Support:**
    ```bash
    python scripts/check_gpu.py
    ```

### GPU Verification Output:
```text
CUDA is available
CUDA version: 12.1
CUDA capabilities: (8, 6)
CUDA devices: 1
CUDA current device: 0
CUDA device name: NVIDIA GeForce RTX 3050 6GB Laptop GPU
CUDA max memory: 6.44 GB
```

> **Note:** Creating a Vision Language Model from scratch requires a CUDA-enabled GPU for efficient training.

