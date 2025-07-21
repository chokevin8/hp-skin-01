#!/bin/bash

# Setup conda environment for hp-skin-01 project

echo "Creating conda environment 'hp-skin-01' with Python 3.10..."

# Create minimal conda environment with just Python
conda create -n hp-skin-01 python=3.10 -y

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hp-skin-01

echo "Installing all packages via pip for faster setup..."

# Upgrade pip first
pip install --upgrade pip

# Install PyTorch with CUDA 11.8
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all Python packages via pip (skip OpenSlide to avoid conda slowness)
echo "Installing Python packages..."
pip install \
    numpy==1.24.3 \
    pandas==2.0.3 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    scikit-image==0.21.0 \
    opencv-python==4.8.0.74 \
    Pillow==10.0.0 \
    tqdm==4.66.1 \
    PyYAML==6.0.1 \
    h5py==3.9.0 \
    ipython==8.14.0 \
    jupyter==1.0.0 \
    segmentation-models-pytorch==0.3.3 \
    albumentations==1.3.1 \
    wandb==0.15.12 \
    torchmetrics==1.2.0 \
    tensorboard==2.14.0

echo "Note: OpenSlide skipped to avoid conda dependency resolution delays."
echo "The code will automatically use PIL fallback for .tif files."

echo "Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "conda activate hp-skin-01"
echo ""
echo "To verify installation, run:"
echo "python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")'"