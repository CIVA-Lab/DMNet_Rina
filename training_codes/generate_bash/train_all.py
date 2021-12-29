

line16="source activate cell"
import os
import os.path as osp
path_dataset="/home/rbync/scratch/cell/"
da_list=os.listdir(path_dataset)

yml_file="../yml/all_eachcell.yml"

branch_mask="mask"
branch_shapemarker="shapemarker"



#if da not in ['BF-C2DL-HSC', 'BF-C2DL-MuSC']:continue
da="all"
for mode in ["GT","ST","GT+ST"]:
    file_all = open(da+mode+".sh", "w")
    if mode=="GT+ST":
        cmd1 = "python ../train_unify_allGTST.py " + da + " " + mode + " " + branch_mask + " " + yml_file + "\n"
        cmd2 = "python ../train_unify_allGTST.py " + da + " " + mode + " " + "center" + " " + "center_ctc" + "\n"

    else:
        cmd1 = "python ../train_unify.py "+da+" "+mode+" "+branch_mask+" "+yml_file+"\n"
        cmd2 = "python ../train_unify.py "+da+" "+mode+" "+branch_shapemarker+" "+yml_file+"\n"


    file_all.write(cmd1)
    file_all.write(cmd2)

    cmd1 = "python ../train_unify.py " + da + " " + mode + " " + branch_mask + " " + yml_file + "\n"
    cmd2 = "python ../train_unify.py " + da + " " + mode + " " + branch_shapemarker + " " + yml_file + "\n"

    file_all.close()



#if da not in ['BF-C2DL-HSC', 'BF-C2DL-MuSC']:continue
da="allBF"
for mode in ["GT","ST","GT+ST"]:
    file_all = open(da+mode+".sh", "w")
    if mode=="GT+ST":
        cmd1 = "python ../train_unify.py " + da + " " + mode + " " + branch_mask + " " + "all_eachcell512" + "\n"
        cmd2 = "python ../train_unify.py " + da + " " + mode + " " + "center" + " " + "center_ctc512" + "\n"

    else:
        cmd1 = "python ../train_unify.py "+da+" "+mode+" "+branch_mask+" "+yml_file+"\n"
        cmd2 = "python ../train_unify.py "+da+" "+mode+" "+branch_shapemarker+" "+yml_file+"\n"


    file_all.write(cmd1)
    file_all.write(cmd2)

    cmd1 = "python ../train_unify.py " + da + " " + mode + " " + branch_mask + " " + yml_file + "\n"
    cmd2 = "python ../train_unify.py " + da + " " + mode + " " + branch_shapemarker + " " + yml_file + "\n"

    file_all.close()

