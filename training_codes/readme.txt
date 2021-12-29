# Author
Rina Bao (ctc682)
rinabao@mail.missouri.edu

This code is for ISBI2021 6th Cell Segmentation and Tracking Challenge secondary track
# Instructions

## Setting up environment

Using Anaconda 3 (or miniconda3) on linux, run the following:

```shell
conda env create -f environment.yml
conda activate cell
```

The Anaconda environment was tested on Linux with CUDA 10.2.

## Training

For six settings in "GT", "ST", "GT+ST", "allGT", "allST", "allGT+allST":
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
