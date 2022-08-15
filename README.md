# DMNet for Cell Segmentation, Detection and Tracking
Accurate segmentation and tracking of cells in microscopy image sequences is extremely beneficial in clinical diagnostic applications and biomedical research. A continuing challenge is the segmentation of dense touching cells and deforming cells with indistinct boundaries,
in low signal-to-noise-ratio images. In this paper, we present a dual-stream marker-guided network (DMNet) for segmentation of touching cells in microscopy videos of many cell types. DMNet uses an explicit cell markerdetection stream, with a separate mask-prediction stream
using a distance map penalty function, which enables supervised training to focus attention on touching and nearby
cells. For multi-object cell tracking we use M2Track tracking-by-detection approach with multi-step data association. Our M2Track with mask overlap includes short term track-to-cell association followed by track-to-track association to re-link tracklets with missing segmentation masks over a short sequence of frames. Our combined detection, segmentation and tracking algorithm has proven its potential on the IEEE ISBI 2021 6th Cell Tracking Challenge (CTC-6) where we achieved multiple top three rankings for diverse cell types.

<p align = "center">
    <img src="/paperimages/workflow.png" alt="DMNet"/>
    <em>Workflow of DMNet and M2Track.</em>
</p>


# Citation

The codes contain the implementations of our below paper, please cite our paper when you are using the codes.

    @inproceedings{bao2021dmnet,
    title={DMNet: Dual-Stream Marker Guided Deep Network for Dense Cell Segmentation and Lineage Tracking},
    author={Bao, Rina and Al-Shakarji, Noor M and Bunyak, Filiz and Palaniappan, Kannappan},
    booktitle={Proceedings of the IEEE International Conference on Computer Vision},
    pages={3361--3370},
    year={2021}
    }

The codes is for ISBI2021 6th Cell Segmentation and Tracking Challenge

# Instructions

## Clone this GitHub repository

```shell
git clone https://github.com/CIVA-Lab/DMNet_Rina.git
```

## Setting up environment

Using Anaconda 3 (or miniconda3) on linux, run the following:

The Anaconda environment was tested on Linux with CUDA 10.2.

```shell
cd training_codes
conda env create -f environment.yml
conda activate cell
```


## Training
 
### Prepare datasets for training

You cand download the training data in the Cell Tracking Challenge webpage http://celltrackingchallenge.net/2d-datasets/.
Download any of the training datasets and place them in the folder `DMNet_Rina/Data/train`. For that, you will need to create the `Data`and `train` folder first.

```shell
mkdir "DMNet_Rina/Data"
mkdir "DMNet_Rina/Data/train"
```

### Download the pretrained model
Download the pretrained HRNet-W32-C model on imagenet from their website (https://github.com/HRNet/HRNet-Image-Classification/) and place them in the folder `models_imagenet`. Use the following code to create the folder:

```shell
cd training_codes
mkdir models_imagenet
```
Once the dataset is placed there, it should look like `DMNet_Rina/Data/train/DIC-C2DH-HeLa`

### Training on all the datasets from the Cell Tracking Challenge
To run the training on all the datasets from the Cell Tracking Challenge, run the six settings "GT", "ST", "GT+ST", "allGT", "allST", "allGT+allST" using the `.bash`codes in the folder `generate_bash`:

```shell
cd generate_bash
bash allGT.sh
bash allST.sh
bash allGT+ST.sh
```
### Training on a specific dataset of the Cell Tracking Challenge

Use the specific name of the Cell Tracking Challenge dataset in the variable `dataset` in the following line:

```shell
bash $dataset.sh
```

### Chaning some training parameters:
Each bash code runs a training `.py`file placed in the `training_codes` folder. The parameters for the training (epochs, batch size, learning rate etc.) are specified in the `yaml` files that you can find in the `yml` folder. Note that you will need to addapt the value of some of these parameters, such as the batch size, according to your hardware (GPU).


## Testing

```shell
cd inference_codes
```

Running all bash files for testing


Thanks!


### Running the code in Google Colab

It's popssible to set up the conda environment in Google Colab using the following:

```shell
!git clone https://github.com/CIVA-Lab/DMNet_Rina.git
!pip install -q condacolab
import condacolab
condacolab.install()
import os
os.chdir("/content/DMNet_Rina/training_codes")
!conda env create -f environment.yml
!source activate cell
```

Relative paths sometimes do not work well in conda, so the only thing you need to do to run the training scripts is to change the relative paths in the `yaml` files that you need to use to the full path. For example, change `../training_codes/models_imagenet` to `/content/DMNet_Rina/training_codes/models_imagenet`.
The same in the corresponding training python script in case there exists any. 

If the training does not work when it is called using `bash`, copy&paste the scripts in the training file directly into the notebook with the correct inputs that are specified in the bash and runn it placing the working directory in `training_codes`. Here is an example:

```python
import multiprocessing
#import pandas as pd
import numpy as np
import skimage
import sys
import os
os.chdir("/content/DMNet_Rina/training_codes/")
sol = __import__('solaris_rina')

# Dataset location (edit as needed)
dataset_namein = "DIC-C2DH-HeLa"
gttype="GT"
branch="mask"
config_path="/content/DMNet_Rina/training_codes/yml/all_eachcell.yml"
# Load config
config = sol.utils.config.parse(config_path)

# %% ============================
# Training
# ===============================

# make model output dir

os.makedirs(os.path.dirname(config['training']['model_dest_path']), exist_ok=True)
config['pretrained_rina']=True
trainer = sol.nets.train_cell_GTST.Trainer(config=config,dataset_name=dataset_namein,branch=branch,GT=gttype)

trainer.train()
```



## Contact Information 
**Created by:** Rina Bao

Department of Electrical Engineering and Computer Science,  
University of Missouri-Columbia  

For more information, contact:

* **Rina Bao**  
University of Missouri-Columbia  
Columbia, MO 65211  
rinabao@mail.missouri.edu  

* **Noor M. Al-Shakarji**  
University of Missouri-Columbia  
Columbia, MO 65211  
nmahyd@missouri.edu

* **Dr. K. Palaniappan**  
205 Naka Hall (EBW)  
University of Missouri-Columbia  
Columbia, MO 65211  
pal@missouri.edu



## Acknowledgement




