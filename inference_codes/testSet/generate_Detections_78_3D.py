import os
import os.path as osp
import skimage.io as sio
import matplotlib.pyplot as plt
import numpy as np
import cv2
from post_processing_shape import postprocess_mask_and_markers,postprocess_mask_and_markers_3d,postprocess_mask_and_watermarkers,simple_label,remove_small_components,regular_norm

#area_thresh_all={"BF-C2DL-MuSC":[0,25],"Fluo-N2DL-HeLa":[0,50],"PhC-C2DH-U373":[0,200],"BF-C2DL-HSC":[0,50],"PhC-C2DL-PSC":[50,50],"Fluo-C2DL-MSC":[0,200],"DIC-C2DH-HeLa":[200,0.5*834],"Fluo-N2DH-GOWT1":[0,138]}

area_thresh_all={"BF-C2DL-MuSC":[0,25],
                 "Fluo-N2DL-HeLa":[0,50],
                 "PhC-C2DH-U373":[0,200],
                 "BF-C2DL-HSC":[0,50],
                 "PhC-C2DL-PSC":[50,25],
                 "Fluo-C2DL-MSC":[0,200],
                 "DIC-C2DH-HeLa":[200,0.5*834],
                 "Fluo-N2DH-GOWT1":[0,138],
                 "Fluo-C3DL-MDA231":[0,50],
                 'Fluo-C3DH-H157':[0,3000],
                 'Fluo-N3DH-CE':[0,200],
                 'Fluo-N3DH-CHO':[0,200],
                 'Fluo-C3DH-A549':[0,300]}


def get_nucleus_ids(img):
    """ Get nucleus ids in intensity-coded label image.

    :param img: Intensity-coded nuclei image.
        :type:
    :return: List of nucleus ids.
    """

    values = np.unique(img)
    values = values[values > 0]

    return values


def foi_correction(mask, cell_type):
    """ Field of interest correction for Cell Tracking Challenge data (see
    https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf and
    https://public.celltrackingchallenge.net/documents/Annotation%20procedure.pdf )

    :param mask: Segmented cells.
        :type mask:
    :param cell_type: Cell Type.
        :type cell_type: str
    :return: FOI corrected segmented cells.
    """

    if cell_type in ['DIC-C2DH-HeLa', 'Fluo-C2DL-Huh7', 'Fluo-C2DL-MSC', 'Fluo-C3DH-H157', 'Fluo-N2DH-GOWT1',
                     'Fluo-N3DH-CE', 'Fluo-N3DH-CHO', 'PhC-C2DH-U373']:
        E = 50
    elif cell_type in ['BF-C2DL-HSC', 'BF-C2DL-MuSC', 'Fluo-C3DL-MDA231', 'Fluo-N2DL-HeLa', 'PhC-C2DL-PSC']:
        E = 25
    else:
        E = 0

    if len(mask.shape) == 2:
        foi = mask[E:mask.shape[0] - E, E:mask.shape[1] - E]
    else:
        foi = mask[:, E:mask.shape[1] - E, E:mask.shape[2] - E]

    ids_foi = get_nucleus_ids(foi)
    ids_prediction = get_nucleus_ids(mask)
    for id_prediction in ids_prediction:
        if id_prediction not in ids_foi:
            mask[mask == id_prediction] = 0

    return mask


def remote_smallmarker(img,thresh):
    cur_labels = sorted(np.unique(img))[1:]

    for cr_l in cur_labels:

        if cr_l == 0:
            continue
        size_obj = np.sum(img == cr_l)
        #if cr_l==12:
        #    print ("check size",size_obj)
        #print (cr_l,size_obj)
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





from Inference_dataset_newmodel_shape3D import Inferer_singleimg
#from submit2_codes.ctc.Inference_dataset_newmodel_center import Inferer_singleimg as Inferer_singleimg_center
import os.path as osp
import glob
sys.path.append("./")

def get_bestmodel(path_use):
    allmodels = glob.glob(osp.join(path_use, "hr_best_epoch*.pth"))
    epoc_min = 0

    for mm in allmodels:
        epo = mm.split("/")[-1].split("hr_best_epoch")[-1].split("_")[0]
        if int(epo) > epoc_min:
            epoc_min = int(epo)
            mask_model_path = mm
    return mask_model_path


def get_detection(input_imgpath,save_rina_seg_path,da,gttype,model_choose="final",vis=0,all=0):
    model_path = "../wdata/ctc/"

    if all:

        if gttype=="allGT+allST":
            path_mask_use = osp.join(model_path, "3Dtrain", "mask", "GT+ST")
            path_center_use = osp.join(model_path, "3Dtrain", "shapemarker", "GT+ST")


        else:
            #path_mask_use = osp.join(model_path, "all", "mask", gttype.split("all")[1])
            #path_center_use = osp.join(model_path, "all", "shapemarker", gttype.split("all")[1])

            path_mask_use = osp.join(model_path, "all", "mask", gttype)
            path_center_use = osp.join(model_path, "all", "shapemarker", gttype)

        mask_model_path = osp.join(path_mask_use, "hrnet_" + model_choose + ".pth")

        # path_center_use=osp.join(model_path,da,"shapemarker",gttype)

        # centermarker_model_path  = osp.join(path_center_use,"hr_"+model_choose+"_shapemarker.pth")

        centermarker_model_path = osp.join(path_center_use, "hrnet_" + model_choose + ".pth")

    else:

        path_mask_use = osp.join(model_path, da, "mask", gttype)

        mask_model_path = osp.join(path_mask_use, "hrnet_" + model_choose + ".pth")

        # path_center_use=osp.join(model_path,da,"shapemarker",gttype)
        path_center_use = osp.join(model_path, da, "shapemarker", gttype)

        # centermarker_model_path  = osp.join(path_center_use,"hr_"+model_choose+"_shapemarker.pth")

        centermarker_model_path = osp.join(path_center_use, "hrnet_" + model_choose + ".pth")

    print ("loading model", mask_model_path,centermarker_model_path)

    center_inferer = Inferer_singleimg(model_path=centermarker_model_path)

    mask_infer = Inferer_singleimg(model_path=mask_model_path)
    path_imglist=(input_imgpath+"/*.tif")
    print (path_imglist)
    imglist=sorted(glob.glob(path_imglist))


    for imgp in imglist:
            print (imgp)
            mask_name="mask"+imgp.split('/')[-1].split('.tif')[0].split('t')[-1]+".tif"
            if da in ['Fluo-C3DL-MDA231','Fluo-N3DH-CE','Fluo-N3DH-CHO']:
                center_pred_all = center_inferer([imgp,da])
            else:
                center_pred_all = center_inferer([imgp,da])

            mask_all = mask_infer([imgp,da])

            center_marker_det_all=np.zeros_like(mask_all,dtype=np.uint16)

            if da in ['Fluo-C3DL-MDA231', 'Fluo-N3DH-CE', 'Fluo-N3DH-CHO', 'Fluo-C3DH-A549', 'Fluo-C3DH-H157']:

              for tt in range(mask_all.shape[0]):
                mask=mask_all[tt,:,:]
                #mask=regular_norm(mask)
                #center_marker_det = simple_label(mask)
                if da in ['Fluo-C3DL-MDA231']:
                    center_pred = center_pred_all[tt, :, :]
                    #center_pred=regular_norm(center_pred)
                    center_marker_det=postprocess_mask_and_watermarkers(mask,center_pred,area_thresh=25)
                elif da in ['Fluo-N3DH-CE','Fluo-N3DH-CHO']:
                    center_pred = center_pred_all[tt, :, :]
                    #center_pred=regular_norm(center_pred)

                    center_marker_det=postprocess_mask_and_markers(mask,center_pred,area_thresh=25)
                else:
                    #da in ['Fluo-C3DH-A549','Fluo-C3DH-H157']:
                    center_marker_det = simple_label(mask)
                    #print (np.unique(center_marker_det))


                center_marker_det=foi_correction(center_marker_det,da)
                center_marker_det_all[tt,:,:]=center_marker_det
                if da in ['Fluo-C3DL-MDA231','Fluo-C3DH-H157','Fluo-N3DH-CE','Fluo-N3DH-CHO', 'Fluo-C3DH-A549']:
                    print ("In remove tiny cell Post Processing", da)
                    center_marker_det = remote_smallmarker(center_marker_det, area_thresh_all[da][1])
                    center_marker_det_all[tt, :, :]=center_marker_det
                vis=0
                if vis:
                    #vis_mask=vis_single(mask)
                    #vis_center=vis_single(center)
                    #vis_img=vis_single(imgori)
                    #showlabel=vis_label(center_marker_det)
                    #cv2.imshow("show",np.concatenate([vis_mask,showlabel],1))
                    #cv2.waitKey()

                    #print (np.unique(center_marker_det))
                    #if np.sum(center_marker_det)>1:

                    plt.imshow(np.concatenate([mask,center_pred,center_marker_det],1))
                    #plt.show()
                    #plt.imshow(center_pred)
                    #plt.show()
                    #plt.imshow(center_marker_det)
                    #plt.show()
                    savepp_t = osp.join(save_rina_seg_path, mask_name+str(tt)+".png")

                    plt.savefig(savepp_t)

            else:
              center_marker_det_all = postprocess_mask_and_markers_3d(mask_all, center_pred_all, area_thresh=25)
              for tt in range(center_marker_det_all.shape[0]):
                        plt.imshow(np.concatenate([mask_all[tt,:,:], center_pred_all[tt,:,:], center_marker_det_all[tt,:,:]], 1))
                        # plt.show()
                        # plt.imshow(center_pred)
                        # plt.show()
                        # plt.imshow(center_marker_det)
                        # plt.show()
                        savepp_t = osp.join(save_rina_seg_path, mask_name + str(tt) + ".png")

                        plt.savefig(savepp_t)

            savepp=osp.join(save_rina_seg_path,mask_name)
            #plt.imshow(center_marker_det)
            #plt.show()
            print ("save at",savepp)
            #print (center_marker_det_all.shape)
            sio.imsave(savepp,center_marker_det_all.astype(np.uint16))










