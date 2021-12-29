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
sum_train=0
sum_test=0
for da in dataset2Dlist:
    print (da)
    list1=get_train_val_dfs(da,"GT")
    print(len(list1[0]), len(list1[1]))
    sum_train=sum_train+len(list1[0])
    sum_test=sum_test+len(list1[1])

print ("sum",sum_train,sum_test)
list1 = get_train_val_dfs("all", "GT")
print(len(list1[0]), len(list1[1]))


data_all_s=[]

from collections import Counter

for im in list1[0]:

    print (im.split("_")[0])
    data_all_s.append(im.split("_")[0])
counter = Counter(data_all_s)
print (counter)

#list1=get_train_val_dfs(da,"ST")
    #print(len(list1[0]), len(list1[1]))

    #list1=get_train_val_dfs(da,"GT+ST")
    #print(len(list1[0]), len(list1[1]))
#
#
# print ("all")
# list1=get_train_val_dfs("all","GT+ST")
# print(len(list1[0]), len(list1[1]))
#
# list1=get_train_val_dfs("all","ST")
# print(len(list1[0]), len(list1[1]))

# list2=get_train_val_dfsrina("BF-C2DL-HSC","GT+ST")
# print (len(list2))
# print (Diff(list1,list2))
# import numpy as np
# data_all=np.load("../data_trainlist/2D3D_cell_allGTallST_train.npy",allow_pickle=True)
# a_list=[]
# data_all_set={}
#
# for da in dataset2Dlist:
#     print (da)
#     data_all_set[da]=[]
#
# for dt in data_all:
#     #print (dt[-1])
#     #a_list.append(dt[-1])
#
#     if dt[-1] not in dataset2Dlist:
#         continue
#     data_all_set[dt[-1]].append(dt[0])
#
#     #print ("aa")
#
# for da in dataset2Dlist:
#     print (da)
#     print (len(np.unique(data_all_set[da])))

#from collections import Counter
#counter = Counter(a_list)

#print (counter)
#for da in dataset2Dlist:


