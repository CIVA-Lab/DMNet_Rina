

line16="source activate cell"
import os
import os.path as osp
path_dataset="/home/rbync/scratch/cell/"
da_list=os.listdir(path_dataset)


branch_mask="mask"
branch_shapemarker="shapemarker"


for da in da_list:

  if da in ['BF-C2DL-HSC', 'BF-C2DL-MuSC']:
      yml_file = "../yml/all_eachcell512.yml"
  else:
      yml_file = "../yml/all_eachcell.yml"

  file_all = open(da+".sh", "w")

  for mode in ["GT","GT+ST","ST"]:


    cmd1 = "python ../train_unify.py "+da+" "+mode+" "+branch_mask+" "+yml_file+"\n"
    cmd2 = "python ../train_unify.py "+da+" "+mode+" "+branch_shapemarker+" "+yml_file+"\n"


    file_all.write(cmd1)
    file_all.write(cmd2)



  file_all.close()
