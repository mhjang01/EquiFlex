#!/bin/bash

set -e

ENV_NAME="flex"

echo ">>> Creating Conda environment: $ENV_NAME"

conda create -n $ENV_NAME python=3.11 -y

source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo ">>> Environment created and activated."
echo ">>> Installing packages..."

# EM; need conda
conda install -c conda-forge openmm[cuda12] -y
conda install -c conda-forge openff-toolkit -y

# Chem
pip install rdkit
pip install py3Dmol
pip install esm
pip install biotite
pip install gemmi
pip install plinder
pip install unimol_tools
pip install huggingface_hub

# ML
pip3 install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install lightning
pip install cuequivariance==0.7.0
pip install cuequivariance-torch==0.7.0
pip install cuequivariance-ops-torch-cu12==0.7.0
pip install pyg_lib -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
#pip install --no-binary :all: torch_cluster -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
#pip install --no-binary :all: torch_sparse -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
#pip install --no-binary :all: torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install torch_geometric
pip install git+https://github.com/KellerJordan/Muon
pip install wandb
pip install torchinfo
pip install transformers
pip install scikit-learn
pip install opt-einsum-fx
pip install hydra-core --upgrade

# Auxilliaries
pip install matplotlib
pip install jupyter
pip install pandas
pip install joblib
pip install seaborn


echo "NEED TO ADJUST PLINDER SYSTEM DIRECTORY at:"
echo "  conda/envs/$ENV_NAME/lib/python3.11/site-packages/plinder/core/utils/config.py"
echo ">>> Installation complete!"