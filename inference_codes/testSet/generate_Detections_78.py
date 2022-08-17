import os
import os.path as osp
import skimage.io as sio
import matplotlib.pyplot as plt
import numpy as np
from post_processing_shape import postprocess_mask_and_markers, postprocess_mask_and_watermarkers, simple_label, \
    remove_small_components

# area_thresh_all={"BF-C2DL-MuSC":[0,25],"Fluo-N2DL-HeLa":[0,50],"PhC-C2DH-U373":[0,200],"BF-C2DL-HSC":[0,50],"PhC-C2DL-PSC":[50,50],"Fluo-C2DL-MSC":[0,200],"DIC-C2DH-HeLa":[200,0.5*834],"Fluo-N2DH-GOWT1":[0,138]}

area_thresh_all = {"BF-C2DL-MuSC": [0, 100],
                   "Fluo-N2DL-HeLa": [0, 50],
                   "PhC-C2DH-U373": [0, 200],
                   "BF-C2DL-HSC": [0, 50],
                   "PhC-C2DL-PSC": [50, 25],
                   "Fluo-C2DL-MSC": [0, 200],
                   "DIC-C2DH-HeLa": [200, 0.5 * 834],
                   "Fluo-N2DH-GOWT1": [0, 200]}


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
    # print ("!!!!!!!!!!!!!!!!",E)
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


def remote_smallmarker(img, thresh):
    cur_labels = sorted(np.unique(img))[1:]

    for cr_l in cur_labels:

        if cr_l == 0:
            continue
        size_obj = np.sum(img == cr_l)
        # if cr_l==12:
        #    print ("check size",size_obj)
        # print (cr_l,size_obj)
        if size_obj < thresh:
            img[img == cr_l] = 0
    return img


def label_renew(img):
    cur_labels = sorted(np.unique(img))[1:]
    max_label_id = 1
    imgnew = np.zeros_like(img)
    for cr_l in cur_labels:

        if cr_l == 0:
            continue
        mask = np.zeros_like(img)

        mask[img == cr_l] = 1

        new_label = simple_label(mask)
        id_new_list = sorted(np.unique(new_label))[1:]
        for id_new in id_new_list:
            # print (max_label_id)
            max_label_id = max_label_id + 1

            imgnew[new_label == id_new] = max_label_id

    return imgnew


import sys

from skimage.morphology import reconstruction
from skimage.morphology import binary_opening

from Inference_dataset_newmodel_shape import Inferer_singleimg
# from submit2_codes.ctc.Inference_dataset_newmodel_center import Inferer_singleimg as Inferer_singleimg_center
import os.path as osp
import glob
from post_processing_shape import contrast_strech_norm, regular_norm

sys.path.append("./")
import cv2


def get_bestmodel(path_use):
    allmodels = glob.glob(osp.join(path_use, "hr_best_epoch*.pth"))
    epoc_min = 0

    for mm in allmodels:
        epo = mm.split("/")[-1].split("hr_best_epoch")[-1].split("_")[0]
        if int(epo) > epoc_min:
            epoc_min = int(epo)
            mask_model_path = mm
    return mask_model_path


def get_detection(input_imgpath, save_rina_seg_path, da, gttype, model_choose="final", vis=0, all=0):
    model_path = "../wdata/ctc/"

    if all:
        if "BF" in da:

            path_mask_use = osp.join(model_path, "allBF", "mask", gttype.split("all")[1])

            mask_model_path = osp.join(path_mask_use, "hrnet_" + model_choose + ".pth")

            # path_center_use=osp.join(model_path,da,"shapemarker",gttype)
            path_center_use = osp.join(model_path, "allBF", "shapemarker", gttype.split("all")[1])

            centermarker_model_path = osp.join(path_center_use, "hrnet_" + model_choose + ".pth")


        else:

            # path_mask_use = osp.join(model_path, "all", "mask", gttype.split("all")[1])
            path_mask_use = osp.join(model_path, "all", "mask", gttype)

            # mask_model_path = osp.join(path_mask_use, "hrnet_" + model_choose + ".pth")
            mask_model_path = osp.join(path_mask_use, "hrnet_" + model_choose + ".pth")

            # path_center_use=osp.join(model_path,da,"shapemarker",gttype)
            # path_center_use = osp.join(model_path, "all", "shapemarker", gttype.split("all")[1])
            path_center_use = osp.join(model_path, "all", "shapemarker", gttype)

            # centermarker_model_path  = osp.join(path_center_use,"hr_"+model_choose+"_shapemarker.pth")

            centermarker_model_path = osp.join(path_center_use, "hrnet_" + model_choose + ".pth")

    else:

        path_mask_use = osp.join(model_path, da, "mask", gttype)

        mask_model_path = osp.join(path_mask_use, "hrnet_" + model_choose + ".pth")

        # path_center_use=osp.join(model_path,da,"shapemarker",gttype)
        path_center_use = osp.join(model_path, da, "shapemarker", gttype)
        print ("check center path", path_center_use)
        # centermarker_model_path  = osp.join(path_center_use,"hr_"+model_choose+"_shapemarker.pth")

        centermarker_model_path = osp.join(path_center_use, "hrnet_" + model_choose + ".pth")

    print ("loading model", mask_model_path, centermarker_model_path)

    center_inferer = Inferer_singleimg(model_path=centermarker_model_path)

    mask_infer = Inferer_singleimg(model_path=mask_model_path)
    path_imglist = (input_imgpath + "/*.tif")
    print (path_imglist)
    imglist = sorted(glob.glob(path_imglist))

    for imgp in imglist:
        # print (imgp)
        mask_name = "mask" + imgp.split('/')[-1].split('.tif')[0].split('t')[-1] + ".tif"

        maskb = mask_infer([imgp, da])
        # mask=contrast_strech_norm(maskb)
        # plt.imshow(np.concatenate([maskb,mask],1))
        # plt.show()
        # center_pred = regular_norm(center_pred)
        # plt.imshow(maskb)
        # plt.savefig(save_rina_seg_path+ mask_name + ".png")

        # plt.show()
        # plt.savefig()
        # print (np.unique(maskb))
        if da in ['DIC-C2DH-HeLa']:
            mask = regular_norm(maskb)
            # center_pred = center_inferer([imgp, da])
            center_pred = center_inferer([imgp, da])
            center_pred = regular_norm(center_pred)
            center_marker_det = postprocess_mask_and_markers(mask, center_pred, area_thresh=area_thresh_all[da][1])

        elif da in ['Fluo-N2DL-HeLa']:
            if gttype in ['GT']:

                mask = regular_norm(maskb)
                center_pred = center_inferer([imgp, da])
                center_pred = regular_norm(center_pred)
                seedimg = 1. * (center_pred > 0.5) * (maskb > 0.5)
                center_marker_mask = reconstruction(seedimg, 1. * (maskb > 0.5), offset=None)
                center_marker_det = postprocess_mask_and_markers(center_marker_mask, seedimg,
                                                                 area_thresh=area_thresh_all[da][1])
            else:
                mask = regular_norm(maskb)

                # center_pred = center_inferer([imgp, da])
                center_pred = center_inferer([imgp, da])
                center_pred = regular_norm(center_pred)
                center_marker_det = postprocess_mask_and_markers(mask, center_pred,
                                                                 area_thresh=area_thresh_all[da][1])

        elif da in ['Fluo-C2DL-MSC']:
            mask = regular_norm(maskb)

            center_marker_det = simple_label(mask)

        elif da in ['PhC-C2DH-U373', ]:
            mask = regular_norm(maskb)

            if gttype in ["ST"]:
                center_pred = center_inferer([imgp, da])
                center_pred = regular_norm(center_pred)
                seedimg = 1. * (center_pred > 0.5) * (maskb > 0.5)
                center_marker_mask = reconstruction(seedimg, 1. * (maskb > 0.5), offset=None)

                center_marker_det = simple_label(center_marker_mask)

            else:
                center_marker_det = simple_label(mask)


        elif da in ['BF-C2DL-MuSC']:
            # if gttype in ['allST','GT+ST','allGT']:
            #    center_pred = center_inferer([imgp, da])

            #    center_marker_mask = reconstruction(1. * (center_pred > 0.5), 1. * (maskb > 0.5), offset=None)

            #    center_marker_det = simple_label(center_marker_mask)
            # else:
            center_pred = center_inferer([imgp, da])
            seedimg = 1. * (center_pred > 0.5) * (maskb > 0.5)
            center_marker_mask = reconstruction(seedimg, 1. * (maskb > 0.5), offset=None)

            center_marker_det = simple_label(center_marker_mask)
            # plt.imshow(np.concatenate([maskb, center_pred, center_marker_mask], 1))

            # plt.show()

            # plt.imshow(reconstruction_mask)
            # plt.show()
        elif da in ['BF-C2DL-HSC']:
            mask = regular_norm(maskb)

            center_pred = center_inferer([imgp, da])
            center_pred = regular_norm(center_pred)

            # reconstruction_mask = reconstruction(1. * (mask > 0.5), 1. * (center_pred > 0.5)+0.5,offset=None)
            # reconstruction_mask = binary_opening(1. * (mask > 0.5), 1. * (center_pred > 0.5))

            # plt.imshow(center_pred)
            # plt.show()

            kernel = np.ones((3, 3), np.uint8)

            center_pred_center = cv2.erode(center_pred.copy(), kernel, iterations=1)

            seedimg = 1. * (center_pred > 0.5) * (mask > 0.5)
            center_marker_mask = reconstruction(seedimg, 1. * (mask > 0.5), offset=None)

            center_marker_det = postprocess_mask_and_watermarkers(center_marker_mask, center_pred_center, area_thresh=5)

            # center_marker_det = cv2.dilate(center_marker_det, kernel, iterations=2)
        elif da in ['Fluo-N2DH-GOWT1']:
            mask = regular_norm(maskb)

            if all:

                # print ("in all mode")
                center_pred = center_inferer([imgp, da])
                center_pred = regular_norm(center_pred)
                kernel = np.ones((5, 5), np.uint8)
                center_pred_center = cv2.erode(center_pred.copy(), kernel, iterations=1)
                rina_ = regular_norm(mask + center_pred)

                center_marker_det = postprocess_mask_and_markers(rina_, center_pred_center, area_thresh=5)
            else:
                if gttype in ["GT"]:
                    center_pred = center_inferer([imgp, da])
                    center_pred = regular_norm(center_pred)
                    kernel = np.ones((5, 5), np.uint8)

                    center_pred_center = cv2.erode(center_pred.copy(), kernel, iterations=1)
                    center_marker_det = postprocess_mask_and_markers(center_pred, center_pred_center, area_thresh=5)
                    # center_marker_det = cv2.dilate(center_marker_det, np.ones((3,3)), iterations=1)

                elif gttype in ["ST"]:
                    center_pred = center_inferer([imgp, da])
                    center_pred = regular_norm(center_pred)
                    # plt.imshow(np.concatenate([mask, center_pred], 1))
                    # plt.show()
                    center_marker_det = postprocess_mask_and_markers(mask, center_pred,
                                                                     area_thresh=area_thresh_all[da][1])

                else:
                    center_marker_det = simple_label(mask)

        elif da in ['PhC-C2DL-PSC']:
            mask = regular_norm(maskb)

            center_pred = center_inferer([imgp, da])
            # center_pred = contrast_strech_norm(center_pred)

            center_marker_det = postprocess_mask_and_watermarkers(mask, center_pred, area_thresh=25)
        else:
            print ("testing on new dataset")
            mask = regular_norm(maskb)

            center_pred = center_inferer([imgp, da])
            center_pred = regular_norm(center_pred)

            center_marker_det = postprocess_mask_and_markers(mask, center_pred,
                                                             area_thresh=50)

        center_marker_det = foi_correction(center_marker_det, da)

        if da in ['PhC-C2DH-U373', 'Fluo-N2DL-HeLa', 'Fluo-C2DL-MSC', 'Fluo-N2DH-GOWT1', 'BF-C2DL-MuSC', 'BF-C2DL-HSC']:
            print ("In remove tiny cell Post Processing", da)
            center_marker_det = remote_smallmarker(center_marker_det, area_thresh_all[da][1])

        # watercenter=simple_label(center_pred*128)
        #
        # if da in ['DIC-C2DH-HeLa']:
        #     watercenter=remote_smallmarker(watercenter,thresh=area_thresh_all[da][0])
        #     print ("In remote tiny marker Post Processing", da)
        #
        # if da in ['BF-C2DL-MuSC','BF-C2DL-HSC','Phc-C2DH-PSC','Fluo-N2DH-GOWT1','Fluo-N2DL-HeLa']:
        #         reconstruction_mask = reconstruction(1. * (center_pred > 0), 1. * (mask > 0)+1. * (center_pred > 0),offset=None)
        #
        #         print ("In Marker Post Processing", da)
        #         center_marker_det=persistence_withmarker(mask,watercenter)
        #         plt.imshow(watercenter)
        #         plt.show()
        #         #center_marker_det = persistence_withmarker(mask, watercenter)
        #
        # else:
        #     # from skimage.morphology import reconstruction
        #     if da in ['PhC-C2DH-U373','Fluo-C2DL-MSC']:
        #         center_marker_det = simple_label(mask)
        #         print ("In Marker Mask Processing", da)
        #
        #
        #     else:
        #         print ("In Marker Mask Processing", da)
        #
        #         center_marker_det = persistence_withmarker(mask, watercenter)
        #         #plt.imshow(center_marker_det)
        #         #plt.show()
        # if da in ['DIC-C2DH-HeLa']:
        #     print ("In remove tiny cell Post Processing", da)
        #     center_marker_det=label_renew(center_marker_det)
        #     center_marker_det=remote_smallmarker(center_marker_det,area_thresh_all[da][1])
        #
        # #plt.imshow(center_marker_det)
        # #plt.show()
        # if da in ['PhC-C2DH-U373','Fluo-N2DL-HeLa','Fluo-C2DL-MSC']:
        #     print ("In remove tiny cell Post Processing", da)
        #     center_marker_det=remote_smallmarker(center_marker_det,area_thresh_all[da][1])
        #
        # if 'BF' in da:
        #     mask = np.zeros_like(center_marker_det)
        #     mask[25:-25, 25:-25] = 1
        #     center_marker_det=center_marker_det*mask
        savepp = osp.join(save_rina_seg_path, mask_name)

        vis = 0
        if vis:
            plt.imshow(np.concatenate([mask, center_pred, center_marker_det], 1))
            # plt.show()
            # plt.imshow(center_marker_det)
            # plt.show()
            plt.savefig(savepp + ".png")
        # plt.imshow(center_marker_det)
        # plt.show()
        print ("save at", savepp)

        sio.imsave(savepp, center_marker_det.astype(np.uint16))










