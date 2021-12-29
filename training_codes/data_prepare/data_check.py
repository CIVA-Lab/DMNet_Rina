import json
def get_train_val_dfs(dataset,gttype):
    pathload = "/home/rbync/codes/cell_organize/training_codes/data_trainlist/ids_" + dataset + "_" + gttype + ".json"
    pathload ="/usr/mvl2/rbync/Rinabackup/DMNet/training_codes/data_trainlist/ids_"+ dataset  + "_"+gttype + ".json"

    print (pathload)
    with open(pathload) as f:
        data_t= json.load(f)

        train_dataset =data_t['train']
        val_dataset = data_t['val']

    return train_dataset,val_dataset

def get_train_val_dfsrina(dataset,gttype):

    pathload = "/home/rbync/codes/cell_organize/training_codes/data_trainlist/"+ dataset + "_" + gttype + ".json"
    pathload ="/usr/mvl2/rbync/Rinabackup/DMNet/training_codes/data_trainlist/"+ dataset  + gttype + ".json"
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
for da in dataset2Dlist:
    print (da)
    list1=get_train_val_dfs(da,"GT")
    print(len(list1[0]), len(list1[1]))

    list1=get_train_val_dfs(da,"ST")
    print(len(list1[0]), len(list1[1]))

    list1=get_train_val_dfs(da,"GT+ST")
    print(len(list1[0]), len(list1[1]))

# list2=get_train_val_dfsrina("BF-C2DL-HSC","GT+ST")
# print (len(list2))
# print (Diff(list1,list2))
