# Author
Rina Bao (ctc682)
rinabao@mail.missouri.edu
The codes contain the implementations of our below paper, please cite our paper when you are using the codes.

@inproceedings{bao2021dmnet,
  title={DMNet: Dual-Stream Marker Guided Deep Network for Dense Cell Segmentation and Lineage Tracking},
  author={Bao, Rina and Al-Shakarji, Noor M and Bunyak, Filiz and Palaniappan, Kannappan},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={3361--3370},
  year={2021}
}

This codes is for ISBI2021 6th Cell Segmentation and Tracking Challenge secondary track
# Instructions

## Clone this GitHub repository

```
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
 
Download the pretrained HRNet-W32-C model on imagenet from their website (https://github.com/HRNet/HRNet-Image-Classification/) and place them in the folder `models_imagenet`. Use the following code to create the folder:

```
cd training_codes
mkdir models_imagenet
```


### Training on all the datasets from the Cell Tracking Challenge
To run the training on all the datasets from the Cell Tracking Challenge, run the six settings "GT", "ST", "GT+ST", "allGT", "allST", "allGT+allST" using the `.bash`codes in the folder `generate_bash`:

```
cd generate_bash
bash allGT.sh
bash allST.sh
bash allGT+ST.sh
```
### Training on a specific dataset of the Cell Tracking Challenge

Use the specific name of the Cell Tracking Challenge dataset in the variable `dataset` in the following line:

```
bash $dataset.sh
```

### Chaning some training parameters:
Each bash code runs a training `.py`file placed in the `training_codes` folder. The parameters for the training (epochs, batch size, learning rate etc.) are specified in the `yaml` files that you can find in the `yml` folder. Note that you will need to addapt the value of some of these parameters, such as the batch size, according to your hardware (GPU).

## Testing

```
cd inference_codes
```

Running all bash files for testing


Thanks!

