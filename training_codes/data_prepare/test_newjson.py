import numpy as np
import json
def get_sub(datasetlist,da_name):
    subset_list=[]
    for da_item in datasetlist:
        if da_name in da_item:
            subset_list.append(da_item)

    return subset_list

import os.path as osp
config={}
config['training_data_names']='/usr/mvl2/rbync/Rinabackup/DMNet/training_codes/data_trainlist'



dataset3Dlist=['Fluo-C3DH-A549','Fluo-C3DL-MDA231','Fluo-C3DH-H157','Fluo-N3DH-CE','Fluo-N3DH-CHO']

def convert_name(dataset):
    da_return=[]
    for dt in dataset:
        #print (dt)
        sp=dt.split("_")

        if sp[0] in dataset3Dlist:
            newsp=sp[0]+"_"+sp[1]+"_"+sp[2]+"_"+sp[3]+"_"+sp[4]
            #print (newsp)

        else:

            newsp=sp[0]+"_"+sp[1]+"_"+sp[2]+"_"+sp[3]
        #print (newsp)
        da_return.append(newsp)
    da_return=np.unique(da_return)
    return da_return


def get_oldjson(dataset,gttype):
    pathload = osp.join("/usr/mvl2/rbync/Rinabackup/top1all/", "ids_all" + gttype + ".json")
    with open(pathload) as f:
        data_t= json.load(f)
        train_dataset =data_t['train']
        val_dataset = data_t['val']
        newtrain_data=[]
        newval_data=[]

        for da in train_dataset.keys():
            newtrain_data.extend(train_dataset[da])
            newtrain_data.extend(val_dataset[da])
            newval_data.extend(val_dataset[da])

    newtrain_data=convert_name(newtrain_data)
    newval_data=convert_name(newval_data)
    return newtrain_data,newval_data


def get_json(dataset,gttype):
    pathload = osp.join("/usr/mvl2/rbync/Rinabackup/top1all/", "ids_all" + gttype + ".json")
    with open(pathload) as f:
        data_t= json.load(f)
        train_dataset =data_t['train']
        val_dataset = data_t['val']


    newtrain_data=convert_name(train_dataset)
    newval_data=convert_name(val_dataset)
    return newtrain_data,newval_data



def get_train_val_dfs(config,dataset,gttype):
    if dataset in ["all","allBF"]:
        pathload = osp.join(config['training_data_names'],"ids_" + dataset + gttype + ".json")

    elif dataset in ["3Dtrain"]:

        pathload = osp.join(config['training_data_names'],"ids_" + dataset + "_" + gttype + ".json")

    else:
        if "BF" in dataset:
            pathload = osp.join(config['training_data_names'],"ids_allBF" + gttype + ".json")

        else:

            pathload = osp.join(config['training_data_names'],"ids_all" + gttype + ".json")

    with open(pathload) as f:
        data_t= json.load(f)
        train_dataset =data_t['train']
        val_dataset = data_t['val']
        #print (val_dataset)
        if dataset not in ["all", "allBF", "3Dtrain"]:


            train_dataset=get_sub(train_dataset,dataset)
            val_dataset=get_sub(val_dataset,dataset)

    return train_dataset, val_dataset

dataset2Dlist=['BF-C2DL-HSC',
               'BF-C2DL-MuSC',
               'DIC-C2DH-HeLa',
               'Fluo-C2DL-MSC',
               'Fluo-N2DH-GOWT1',
               'Fluo-N2DL-HeLa',
               'PhC-C2DH-U373',
               'PhC-C2DL-PSC']
#for da in dataset2Dlist:
#    trainlist,vallist=get_train_val_dfs(config,da,"ST")

#    print (da,len(trainlist),len(vallist))

train_da,val_da=get_json("all","GT+allST")
print (len(train_da),len(val_da))
gttype="ST"
train_da,val_da=get_oldjson("all",gttype)
print (len(train_da),len(val_da))


data_a={}
data_a['train']=train_da
data_a['val']=val_da
np.save("ids_all"+gttype,data_a)

#print (val_da)