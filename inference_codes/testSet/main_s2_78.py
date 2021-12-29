

import sys
sys.path.append("./testSet/")
from generate_Detections_78 import get_detection
#from track_bbox import get_noor_track_results as get_noor_track_results_bbox
#from track_mask import get_noor_track_results as get_noor_track_results_mask

import os
import os.path as osp
from post_processing_shape import roi_correct
BF_list=["BF-C2DL-HSC",
            "BF-C2DL-MuSC"]
def main(inp):

    da=inp[1]
    se=inp[2]
    out_path=inp[3]

    da_name=inp[4]
    gttype=inp[5]
    all=int(inp[6])
    ################ generating detection #################
    input_imgpath=osp.join(da,se)
    det_path=out_path
    os.makedirs(det_path, exist_ok=True)
    print (det_path)
    get_detection(input_imgpath, det_path, da_name,gttype,vis=0,all=all)
    #get_noor_track_results(inp)
    ################ generating tracking ##################
    #if da in ['BF-C2DL-MuSC', 'Fluo-C2DL-MSC']:
    #    get_noor_track_results_mask(det_path,da,se,out_path)
    #else:
    #    get_noor_track_results_bbox(det_path,da,se,out_path)

    if da_name in BF_list:
        roi_correct(det_path, det_path)

if __name__ == '__main__':
    in_p=sys.argv
    main(in_p)
