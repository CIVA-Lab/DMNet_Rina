
import os
import os.path as osp
path_dataset="/home/rbync/scratch/cell/"
da_list=os.listdir(path_dataset)


path_save_local="/home/rbync/codes/cell_ctc/out_submit_test_regular/"
os.makedirs(path_save_local,exist_ok=True)


path_in="../"
path_out="../"


da_2D_list=[
            "DIC-C2DH-HeLa",
            "Fluo-N2DL-HeLa",
            "PhC-C2DL-PSC",
            "Fluo-C2DL-MSC",
            "Fluo-N2DH-GOWT1",
            "PhC-C2DH-U373",
            "BF-C2DL-HSC",
            "BF-C2DL-MuSC",]

da_3D_list=["Fluo-C3DH-A549",
            "Fluo-C3DL-MDA231",
            "Fluo-C3DH-H157",
            "Fluo-N3DH-CE",
            "Fluo-N3DH-CHO",
            ]

for mode in ["allGT+allST"]:
    file_all = open(mode + ".sh", "w")

    for da in da_list:
      path_save = osp.join(path_save_local, da, mode)
      os.makedirs(osp.join(path_save_local, da), exist_ok=True)
      os.makedirs(osp.join(path_save_local, da, mode), exist_ok=True)
      os.makedirs(path_save, exist_ok=True)
      namejob = da + mode
      for seq in ["01", "02"]:
        inputpath = path_in + da
        outputpath= path_out + da + "/" + seq + "_RES-" + mode
        if "all" in mode:
            all=1
        else:
            all=0
        if da in da_3D_list:

             cmd = "python ../main_s2_78_3D.py " + inputpath + " " + seq + " " + outputpath+ " " + da + " " + mode + " "+str(all)+ "\n"
             file_all.write(cmd)

    file_all.close()
