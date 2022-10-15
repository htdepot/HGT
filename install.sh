#!/usr/bin/env bash
# install requirements
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
pip install torch-scatter==2.0.7 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu102.html
pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu102.html
pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu102.html
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu102.html
pip install torch-geometric==2.0.4
pip install timm==0.5.4
pip install scikit-image==0.19.2
pip install nibabel==3.2.2
pip install tqdm==4.63.0