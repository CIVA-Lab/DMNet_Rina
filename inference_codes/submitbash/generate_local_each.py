
import os
import os.path as osp
path_dataset="/home/rbync/scratch/cell/"




path_in=path_dataset
path_out="/home/rbync/scratch/wdata/sub78_verifycodes/"

#path_in="../"
#path_out="../"
da_list=["DIC-C2DH-HeLa",
            "Fluo-N2DL-HeLa",
            "PhC-C2DL-PSC",
            "Fluo-C2DL-MSC",
            "Fluo-N2DH-GOWT1",
            "PhC-C2DH-U373",
            "BF-C2DL-HSC",
            "BF-C2DL-MuSC",
            "Fluo-C3DH-A549",
            "Fluo-C3DL-MDA231",
            "Fluo-C3DH-H157",
            "Fluo-N3DH-CE",
            "Fluo-N3DH-CHO"]

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

for mode in ["ST", "GT", "GT+ST","allGT","allST","allGT+allST"]:

    for da in da_list:

      for seq in ["01", "02"]:

        file_each = open(da+"-"+seq+"-"+mode + ".sh", "w")

        inputpath = path_in + da +"/test/"+da
        outputpath= path_out + da + "/" + seq + "_RES-" + mode
        if "all" in mode:
            all=1
        else:
            all=0



        if da in da_2D_list:
           if mode in ["allGT+allST"]:
               if da in ["PhC-C2DL-PSC"]:
                    cmd = "python ./testGTST/main_s2_psc.py " + inputpath + " " + seq + " " + outputpath + " " + da + " " + mode + "\n"
               else:
                    cmd = "python ./testGTST/main_s2.py " + inputpath + " " + seq + " " + outputpath + " " + da + " " + mode + "\n"

               file_each.write(cmd)
           else:

              cmd = "python ./testSet/main_s2_78.py " + inputpath + " " + seq + " " + outputpath+ " " + da + " " + mode + " "+str(all)+ "\n"
              file_each.write(cmd)


        else:

             cmd = "python ./testSet/main_s2_78_3D.py " + inputpath + " " + seq + " " + outputpath+ " " + da + " " + mode + " "+str(all)+ "\n"
             file_each.write(cmd)



        file_each.close()