# Author
Rina Bao (ctc682)
rinabao@mail.missouri.edu

This code is for ISBI2021 6th Cell Segmentation and Tracking Challenge secondary track
# Instructions

## Setting up environment

Using Anaconda 3 (or miniconda3) on linux, run the following:

```shell
conda env create -f environment.yml
conda activate cell_test_rina
```

The Anaconda environment was tested on Linux with CUDA 10.2.

## Running the code

For five settings in "GT", "ST", "GT+ST", "allGT", "allST":

cd submitbash
bash GT.sh
bash ST.sh
bash GT+ST.sh
bash allGT.sh
bash allST.sh


For setting "allGT+ST":
For 3D datasets: 
bash allGT+ST.sh
For 2D datasets:
cd CTB
run $dataset.sh



Thanks!
