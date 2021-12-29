import numpy as np
from collections import Counter
import json
def get_train_val_dfs(dataset,gttype):
    pathload = "/usr/mvl2/rbync/Rinabackup/DMNet/training_codes/data_trainlist/ids_" + dataset + gttype + ".json"

    print (pathload)
    with open(pathload) as f:
        data_t= json.load(f)

        train_dataset =data_t['train']
        val_dataset = data_t['val']

    return train_dataset,val_dataset

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


da='all'

#list1=get_train_val_dfs(da,"ST")

def get_sub(datasetlist,da_name):
    subset_list=[]
    for da_item in datasetlist:
        if da_name in da_item:
            subset_list.append(da_item)

    return subset_list
list1=np.load("../data_trainlist/ids_allGT.npy",allow_pickle=True)
print (len(list1.item()['train']),len(list1.item()['val']))
sub_=get_sub(list1.item()['val'],'BF-C2DL-MuSC')
print (len(sub_))