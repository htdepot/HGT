# Inferring Tissue Microstructure from Undersampled Diffusion MRI via a Hybrid Graph Transformer
## Brief
This is an implementation of **Inferring Tissue Microstructure from Undersampled Diffusion MRI via a Hybrid Graph Transformer** by **Pytorch**.

In this work, we jointly consider the information in both x-space and q-space, overcoming the limitations of existing methods that are unable to make full use of joint x-q space information. The highlights of our work lie in three-fold:

- We propose a hybrid graph transformer (HGT) to jointly consider the information in both x-space and q-space for improving the accuracy of microstructural estimation.
- Our HGT is the first transformer dedicated to microstructure estimation with an improved architecture equipped with residual and dense connections.
- Extensive experiments on data from the [Human Connectome Project](https://db.humanconnectome.org/) demonstrate the advantages of our HGT over cutting-edge models.

## Model
<img src="./misc/model.png" alt="show" style="zoom:90%;" />
An overview of HGT. The model consists of two modules: q-space learning with a GNN and x-space learning with a transformer. RDT: Residual Dense Transformer; TransLayer: Transformer layer; SRA: Spatial-Reduction Attention.

## Results
We trained the network with an NVIDIA GeForce GTX 2080 GPU with 8GB RAM.

Quantitative evaluation of NODDI indices using PSNR, SSIM, and NRMSE for single-shell undersampled data (30 gradient directions total for b=1000 s/mm2). The best results are in **bold**.
<img src="./misc/result_noddi_30.png" alt="show" style="zoom:90%;" />

Quantitative evaluation of DKI indices using PSNR, SSIM, and NRMSE for single-shell undersampled data (30 gradient directions total for b=1000 s/mm2). The best results are in **bold**.
<img src="./misc/result_dki_30.png" alt="show" style="zoom:90%;" />

## Usage
### Environment
```python
pip install -r requirement.txt
```
If you are installing in a linux environment, you can run the following actions.
```bash
bash install.sh
```
### Data Preparation

First, you should organize the data as follows:

```shell 
data/
├── 100610
    ├── data.nii.gz # HCP data file
    ├── nodif_brain_mask.nii.gz # mask file(you can use dipy to generate)
    ├── bvec # b-value data file
    └── bval # b-value data direction file
├── 102311 
    ├── data.nii.gz
    ├── nodif_brain_mask.nii.gz
    ├── bvec
    └── bval
├── bvec 
└── bval
```

 Second, you can run `prepare_data.py` to process the data:

```python
python prepare_data.py  --path [dataset root]
```

### Training

```python
# To train the DKI model you only need to change the microstructure_name
python train.py --config './config/hgt_config.py' --microstructure_name 'NODDI'
```

### Test/Evaluation

```python
# To train the DKI model you only need to change the microstructure_name
# If you do not want to generate a prediction file just change --is_generate_image to False
python test.py --config './config/hgt_config.py' --microstructure_name 'NODDI' --is_generate_image True
```
### Trianed Parameter

[Predicted NODDI using 30 gradient directions dMRI](https://drive.google.com/file/d/14_GE-ijcWq4xKWS2g0hkbdpHmbO9omYV/view?usp=drive_link)

## Acknowledge

We implment the code by referring to the following projects:

- https://github.com/pyg-team/pytorch_geometric
- https://github.com/4uiiurz1/pytorch-nested-unet
- https://github.com/whai362/PVT

