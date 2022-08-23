
# Colab tutorial

We provide a colab running tutorial. Please read and run the DMNet_colab.ipynb.


There are two streams in the DMNet: mask stream and marker stream

## Train the mask stream

python ../train_unify.py DIC-C2DH-HeLa GT mask ../yml/all_eachcell.yml


## Train the marker stream. 
There are two types of markers, center marker vs shape marker. You can choose based on your available data

python ../train_unify.py DIC-C2DH-HeLa GT shapemarker ../yml/all_eachcell.yml

python ../train_unify.py DIC-C2DH-HeLa GT center ../yml/all_eachcell.yml

