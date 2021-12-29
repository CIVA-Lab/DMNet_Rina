

import os
import os.path as osp
path_dataset="/home/rbync/scratch/cell/"
path_dataset="/usr/mvl5/Images2/BIO2/cell_tracking_challenge/CellTracking_allanno/"
path_save="/usr/mvl2/rbync/Rinabackup/DMNet/training_codes/Data/test/"
path_save="/usr/mvl2/rbync/Rinabackup/DMNet/"

da_list=  ['BF-C2DL-HSC', 'BF-C2DL-MuSC', 'DIC-C2DH-HeLa', 'Fluo-C2DL-MSC', 'Fluo-N2DH-GOWT1', 'Fluo-N2DL-HeLa', 'PhC-C2DH-U373' ,'PhC-C2DL-PSC']
da_list=os.listdir(path_dataset)
file_all = open("creatsofttest.sh", "w")

for da in da_list:


    cmd = "ln -s "+path_dataset+da+"/test/" +da+" "+path_save+da+"\n"


    file_all.write(cmd)



file_all.close()
