import os
import os.path as osp
import skimage.io as sio
import numpy as np
from post_processing import persistence_withmarker,postprocess_mask_and_markers,persistence,simple_label



area_thresh_all={"BF-C2DL-MuSC":[0,25],"Fluo-N2DL-HeLa":[0,50],"PhC-C2DH-U373":[0,200],"BF-C2DL-HSC":[0,50],"PhC-C2DL-PSC":[50,50],"Fluo-C2DL-MSC":[0,200],"DIC-C2DH-HeLa":[200,0.5*834],"Fluo-N2DH-GOWT1":[0,138]}


def remote_smallmarker(img,thresh):
    cur_labels = sorted(np.unique(img))[1:]

    for cr_l in cur_labels:

        if cr_l == 0:
            continue
        size_obj = np.sum(img == cr_l)
        #if cr_l==12:
        #    print ("check size",size_obj)
        #print (size_obj)
        if size_obj<thresh:
            img[img==cr_l]=0
    return img


def label_renew(img):
    cur_labels = sorted(np.unique(img))[1:]
    max_label_id=1
    imgnew=np.zeros_like(img)
    for cr_l in cur_labels:

        if cr_l == 0:
            continue
        mask = np.zeros_like(img)

        mask[img==cr_l]=1

        new_label=simple_label(mask)
        id_new_list=sorted(np.unique(new_label))[1:]
        for id_new in id_new_list:
            #print (max_label_id)
            max_label_id=max_label_id+1

            imgnew[new_label==id_new]=max_label_id

    return imgnew

import sys

from skimage.morphology import reconstruction





from Inference_dataset_newmodel import Inferer_singleimg
from Inference_dataset_newmodel_center_psc import Inferer_singleimg as Inferer_singleimg_center

import glob
sys.path.append("../")



def get_detection(input_imgpath,save_rina_seg_path,da):

    if "BF" in da:
        mask_model_path = "../wdata/ctc/allBF/mask/GT+ST/hrnet_final.pth"
        centermarker_model_path = "../wdata/ctc/allBF/center/GT+ST/hrnet_final.pth"
    else:

        centermarker_model_path = "../wdata/ctc/all/center/GT+ST/hrnet_final.pth"

        mask_model_path = "../wdata/ctc/all/mask/GT+ST/hrnet_final.pth"

    print ("loading model", mask_model_path,centermarker_model_path)


    center_inferer = Inferer_singleimg_center(model_path=centermarker_model_path)

    mask_infer = Inferer_singleimg(model_path=mask_model_path)

    imglist=sorted(glob.glob(input_imgpath+"/*.tif"))


    for imgp in imglist:
            print (imgp)
            mask_name="mask"+imgp.split('/')[-1].split('.tif')[0].split('t')[-1]+".tif"

            center_pred = center_inferer([imgp,da])

            mask = mask_infer([imgp,da])

            watercenter=persistence(center_pred)

            if da in ['DIC-C2DH-HeLa']:
                watercenter=remote_smallmarker(watercenter,thresh=area_thresh_all[da][0])
                print ("In remote tiny marker Post Processing", da)

            if da in ['BF-C2DL-MuSC','BF-C2DL-HSC','Phc-C2DH-PSC','Fluo-N2DH-GOWT1','Fluo-N2DL-HeLa']:
                    reconstruction_mask = reconstruction(
                    1. * (center_pred > 0), 1. * (mask > 0)+1. * (center_pred > 0),
                    offset=None)

                    print ("In Marker Post Processing", da)
                    center_marker_det=persistence_withmarker(reconstruction_mask,watercenter)

            else:
                # from skimage.morphology import reconstruction
                if da in ['PhC-C2DH-U373','Fluo-C2DL-MSC']:
                    center_marker_det = simple_label(mask)
                    print ("In Marker Mask Processing", da)

                    #plt.imshow(center_marker_det)
                    #plt.show()
                else:
                    print ("In Marker Mask Processing", da)

                    center_marker_det = persistence_withmarker(mask, watercenter)

            if da in ['DIC-C2DH-HeLa']:
                print ("In remove tiny cell Post Processing", da)
                center_marker_det=label_renew(center_marker_det)
                center_marker_det=remote_smallmarker(center_marker_det,area_thresh_all[da][1])

            #plt.imshow(center_marker_det)
            #plt.show()
            if da in ['PhC-C2DH-U373','Fluo-N2DL-HeLa','Fluo-C2DL-MSC']:
                print ("In remove tiny cell Post Processing", da)
                center_marker_det=remote_smallmarker(center_marker_det,area_thresh_all[da][1])

            if 'BF' in da:
                mask = np.zeros_like(center_marker_det)
                mask[25:-25, 25:-25] = 1
                center_marker_det=center_marker_det*mask

            savepp=osp.join(save_rina_seg_path,mask_name)
            #plt.imshow(center_marker_det)
            #plt.show()
            print ("svae at",savepp)

            sio.imsave(savepp,center_marker_det.astype(np.uint16))










