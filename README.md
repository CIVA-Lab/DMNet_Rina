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

## Setting up environment

Using Anaconda 3 (or miniconda3) on linux, run the following:

The Anaconda environment was tested on Linux with CUDA 10.2.

```shell
conda env create -f environment.yml
conda activate cell
```


## Training
1> Download the pretrained HRNet on imagenet from their website.

mkdir models_imagenet

https://github.com/HRNet/HRNet-Image-Classification/

download the HRNet-W32-C model and put the model in models_imagenet

2> For six settings in "GT", "ST", "GT+ST", "allGT", "allST", "allGT+allST":
cd training_codes
cd generate_bash
bash allGT.sh
bash allST.sh
bash allGT+ST.sh

For each dataset configuration training,
bash $dataset.sh

## Testing

cd inference_codes

Running all bash files for testing


Thanks!

