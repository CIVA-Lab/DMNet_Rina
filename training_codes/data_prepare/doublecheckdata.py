import numpy as np
import os.path as osp

dataset3Dlist=['Fluo-C3DH-A549','Fluo-C3DL-MDA231','Fluo-C3DH-H157','Fluo-N3DH-CE','Fluo-N3DH-CHO']
dataset2Dlist=['BF-C2DL-HSC',
               'BF-C2DL-MuSC',
               'DIC-C2DH-HeLa',
               'Fluo-C2DL-MSC',
               'Fluo-N2DH-GOWT1',
               'Fluo-N2DL-HeLa',
               'PhC-C2DH-U373',
               'PhC-C2DL-PSC']
dataset2DlistnoBF=[
               'DIC-C2DH-HeLa',
               'Fluo-C2DL-MSC',
               'Fluo-N2DH-GOWT1',
               'Fluo-N2DL-HeLa',
               'PhC-C2DH-U373',
               'PhC-C2DL-PSC']


def get_sub(datasetlist,da_name):
    subset_list=[]
    for da_item in datasetlist:
        if da_name in da_item:
            subset_list.append(da_item)

    return subset_list


def get_train_val_dfs(config,dataset,gttype):
    pathload = osp.join(config['training_data_names'], "ids_all" + gttype + ".npy")

    data_all=np.load(pathload,allow_pickle=True).item()


    if dataset =="allBF":
        dalist=['BF-C2DL-HSC',
               'BF-C2DL-MuSC']
        sub_train=[]
        sub_val=[]
        for da in dalist:
            sub_train_da = get_sub(data_all['train'], da)
            sub_val_da = get_sub(data_all['val'], da)

            sub_train.extend(sub_train_da)
            sub_val.extend(sub_val_da)


    elif dataset == "3Dtrain":

        dalist = ['Fluo-C3DH-A549', 'Fluo-C3DL-MDA231', 'Fluo-C3DH-H157', 'Fluo-N3DH-CE', 'Fluo-N3DH-CHO']

    elif dataset=="all":
        dalist=[]
        dalist.extend(dataset3Dlist)
        dalist.extend(dataset2DlistnoBF)

    else:
        dalist=[dataset]

    sub_train = []
    sub_val = []

    for da in dalist:
        sub_train_da = get_sub(data_all['train'], da)
        sub_val_da = get_sub(data_all['val'], da)

        sub_train.extend(sub_train_da)
        sub_val.extend(sub_val_da)

    return sub_train,sub_val
config={}
config['training_data_names']="../data_trainlist/"
train_sub,val_sub=get_train_val_dfs(config,"all","ST")
#print (train_sub)
print (len((train_sub)),len((val_sub)))
