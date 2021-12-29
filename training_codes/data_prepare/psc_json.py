import json
def get_train_val_dfs(dataset,gttype):
    pathload = "/home/rbync/codes/cell_organize/training_codes/data_trainlist/ids_" + dataset + "_" + gttype + ".json"
    if dataset in ["all"]:
        pathload = "/usr/mvl2/rbync/Rinabackup/DMNet/training_codes/data_trainlist/ids_" + dataset + gttype + ".json"

    else:
        pathload ="/usr/mvl2/rbync/Rinabackup/DMNet/training_codes/data_trainlist/ids_"+ dataset  + "_"+gttype + ".json"

    print (pathload)
    with open(pathload) as f:
        data_t= json.load(f)

        train_dataset =data_t['train']
        val_dataset = data_t['val']

    return train_dataset,val_dataset

def get_train_val_dfsrina(dataset,gttype):

    pathload = "/home/rbync/codes/cell_organize/training_codes/data_trainlist/"+ dataset + "_" + gttype + ".json"
    pathload ="/usr/mvl2/rbync/Rinabackup/top1splits/splits/"+ dataset  + gttype + ".json"
    print (pathload)
    with open(pathload) as f:
        data_t= json.load(f)


        train_dataset =data_t['train']
        val_dataset = data_t['val']

    return ((train_dataset))
def Diff(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))

#list1=get_train_val_dfs("BF-C2DL-HSC","GT")
dataset2Dlist=['BF-C2DL-HSC',
               'BF-C2DL-MuSC',
               'DIC-C2DH-HeLa',
               'Fluo-C2DL-MSC',
               'Fluo-N2DH-GOWT1',
               'Fluo-N2DL-HeLa',
               'PhC-C2DH-U373',
               'PhC-C2DL-PSC']

from collections import Counter

da='3Dtrain'
import numpy as np
list1=get_train_val_dfs(da,"GT+ST")
print (len(list1[0]),len(list1[1]))

print (len(np.unique(list1[0])),len(np.unique(list1[1])))