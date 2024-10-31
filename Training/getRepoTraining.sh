#!/bin/bash
#get clone of repo from github
# Usage: getRepo.sh <dataset> <DOWNLOAD_PATH>
# <dataset> : dataset to be used
# <DOWNLOAD_PATH> : path to download the dataset
# Example: getRepo.sh
git clone https://github.com/mlcommons/training.git

if command -v pip3 &>/dev/null; then
    echo "pip3 is already installed."
else
    echo "pip3 not found, installing locally..."

    # Download get-pip.py
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

    # Install pip locally without sudo
    python3 get-pip.py --user

    # Clean up
    rm get-pip.py

    echo "pip3 installed in user directory."
    echo "You may need to add \$HOME/.local/bin to your PATH."
fi
export PATH="$HOME/.local/bin:$PATH"
DOWNLOAD_PATH=$2
#if dataset is SSD, download the dataset
if [ "$1" == "SSD" ]; then
    cd ./single_stage_detector
    # download all requirements scikit-image>=0.15.0 ujson>=4.0.2 matplotlib>=3.5.1 pycocotools>=2.0.4 git+https://github.com/mlcommons/logging.git@1.1.0-rc4 fiftyone==0.15.1
    pip install -r requirements.txt
    cd ./scripts
    pip install fiftyone
    ./download_openimages_mlperf.sh -d $DOWNLOAD_PATH
    ./download_backbone.sh

    
fi
