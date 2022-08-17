import cv2
import numpy as np
import torch
from skimage import measure, morphology, segmentation
from skimage.morphology import extrema, h_maxima, reconstruction, local_maxima, thin
from scipy import ndimage
import matplotlib.pyplot as plt





def remove_small_components(tensor, area_thresh=200):
    out = (tensor > 0.5).astype('uint8') * 255

    nc, cc, stats, _ = cv2.connectedComponentsWithStats(out)
    for i, stat in enumerate(stats):
        area = stat[-1]
        if area < area_thresh:
            cc[cc == i] = 0

    cc = (cc > 0).astype('float32')

    out = (cc < .5).astype('uint8') * 255
    nc, cc, stats, _ = cv2.connectedComponentsWithStats(out)
    for i, stat in enumerate(stats):
        area = stat[-1]
        if area < area_thresh:
            cc[cc == i] = 0

    cc = (cc < 0.5).astype('float32')
    return cc
from skimage.measure import label


def simple_label(unet_output):
    labels, nb = ndimage.label(unet_output*50-25 > 0)
    return labels

def persistence(unet_output):
    sum_img = np.asarray(unet_output)
    h = 2  ############ increase the height ######### merging criteria

    ######## try h=5 ##########
    ############### change to extended maxima #############

    ############### local maxima is implemented by Yangyang to match the extend_maxima function in matlab

    h_maxima_output = reconstruction(
        sum_img - h, sum_img, method='dilation', selem=np.ones((3, 3), dtype=int), offset=None)
    region_max = local_maxima(h_maxima_output, connectivity=2)
    label_h_maxima = label(region_max, connectivity=2)
    # use peaks and summed images to get watershed separation line
    labels = segmentation.watershed(-sum_img, label_h_maxima,
                       watershed_line=True, connectivity=2)
    split_line = labels == 0

    split_line = split_line.astype(int)

    watershed_label = label((1 - split_line), connectivity=2)

    # split_line = thin(split_line)
    split_line = np.where(sum_img == 0, 0, split_line)

    split_img = sum_img > 0
    split_img = split_img.astype(int)
    split_img = np.where(split_line == 1, 0, split_img)
    split_img = split_img * watershed_label

    return split_img

def postprocess(pred, area_thresh=200):
    pred = remove_small_components(pred)
    _, distance = morphology.medial_axis(pred > 0, return_distance=True)
    maxima = morphology.dilation(morphology.local_maxima(distance), np.ones((5, 5)))
    markers = measure.label(maxima)
    labels = segmentation.watershed(-distance, markers, mask=pred)
    for i in range(1, labels.max() + 1):
        if np.sum(labels == i) < area_thresh:
            labels[labels == i] = 0
    return labels




def persistence_withmarker(unet_output,marker):
    sum_img = np.asarray(unet_output)


    labels = segmentation.watershed(-sum_img, marker,
                       watershed_line=True, connectivity=2)
    split_line = labels == 0

    split_line = split_line.astype(int)

    watershed_label = label((1 - split_line), connectivity=2)

    # split_line = thin(split_line)
    split_line = np.where(sum_img == 0, 0, split_line)

    split_img = sum_img > 0
    split_img = split_img.astype(int)
    #split_img = np.where(split_line == 1, 0, split_img)
    split_img = split_img * watershed_label

    return split_img

from skimage.morphology import binary_closing

def postprocess_mask_and_markers(mask, markers_ori, area_thresh=50,closing=False):
    # Label markers if not labeled

    markers = measure.label(markers_ori.copy()>0.5)
    # Remove small components
    mask = remove_small_components(mask, area_thresh=area_thresh)
    if closing:
        mask=1.0*mask>0.0
        kernel = np.ones(shape=(5, 5))
        mask_new = binary_closing(mask, kernel)
        #plt.imshow(np.concatenate([mask,mask_new],1))
        #plt.show()
    # Label markers if not labeled
    # Calculate distance transform for watershed
    _, distance = morphology.medial_axis(mask>0.,return_distance=True)
    #plt.imshow(markers)
    #plt.show()

    #_, distance_markers = morphology.medial_axis(markers > 0., return_distance=True)
    #plt.imshow(water_marker)
    #plt.show()
    # Use watershed
    labels = segmentation.watershed(-distance, markers, mask=mask)

    # remove small components
    for i in range(1, labels.max() + 1):
        #print (np.sum(labels == i))
        if np.sum(labels == i) < area_thresh:
            labels[labels == i] = 0

    return labels






def watershed(pred,area_thresh):
    #pred = remove_small_components(pred,area_thresh)
    _, distance = morphology.medial_axis(pred > 0.5, return_distance=True)
    maxima =morphology.local_maxima(distance)
    plt.imshow(maxima)
    plt.show()
    markers = measure.label(maxima)
    labels = segmentation.watershed(-distance, markers, mask=pred)
    return labels

import cc3d
import scipy.ndimage as ndi

def remove_small_components_label(label, area_thresh=200):
    label = cc3d.connected_components(label > 0.5, connectivity=6)

    for i in range(1, label.max() + 1):
        if np.sum(label == i) < area_thresh:
            label[label == i] = 0
    return label > 0

def postprocess_mask_and_markers_3d(mask, markers, area_thresh=50):
    # Label markers if not labeled
    markers = ndi.median_filter(markers, footprint=np.ones((5, 1, 1)))
    markers = cc3d.connected_components(markers > 0.5, connectivity=6)

    # Remove small components
    mask = remove_small_components_label(mask, area_thresh=area_thresh)

    # Calculate distance transform for watershed
    distance = ndi.morphology.distance_transform_edt(mask > 0,
                                                     return_distances=True, return_indices=False)

    # Use watershed
    distance = ((1 - distance) * 255).astype('uint8')
    labels = ndi.watershed_ift(distance, markers.astype('int')) * mask

    # remove small components
    for i in range(1, labels.max() + 1):
        if np.sum(labels == i) < area_thresh:
            labels[labels == i] = 0

    return labels


def postprocess_mask_and_watermarkers(mask, markers_ori, area_thresh=50):
    # Label markers if not labeled
    #markers = measure.label(markers > 0.5)
    # Remove small components
    mask = remove_small_components(mask, area_thresh=area_thresh)

    # Calculate distance transform for watershed
    _, distance = morphology.medial_axis(mask > 0., return_distance=True)

    #water_marker=watershed(markers_ori,area_thresh)
    water_marker=persistence(markers_ori*50.0-25)

    #plt.imshow(water_marker)
    #plt.show()


    # Use watershed
    labels = segmentation.watershed(-distance, water_marker, mask=mask)

    # remove small components
    for i in range(1, labels.max() + 1):
        if np.sum(labels == i) < area_thresh:
            labels[labels == i] = 0

    return labels



from scipy.stats import stats

def contrast_strech(img):
    imgori=img.copy()
    img=img.astype(np.float32)

    imgs = img.flatten()

    z = np.abs(stats.zscore(imgs))
    threshold = 2.5

    imgs = imgs[np.where(z <= threshold)]
    norm_v=(np.max(imgs) - np.min(imgs))
    if norm_v>0:
        imgnew = (img - np.min(imgs)) / norm_v
        #print (np.min(imgnew),np.max(imgnew))
        imgnew[imgnew <=0] = 0
        imgnew[imgnew >= 1] = 1
        imgnew=imgnew * 255
    else:
        imgnew=imgori
    imgnew=np.asarray(imgnew,dtype=np.uint8)
    return imgnew


def contrast_strech_norm(img):
    imgori=img.copy()
    img=img.astype(np.float32)

    imgs = img.flatten()

    z = np.abs(stats.zscore(imgs))
    threshold = 2.5

    imgs = imgs[np.where(z <= threshold)]
    norm_v=(np.max(imgs) - np.min(imgs))
    if norm_v>0:
        imgnew = (img - np.min(imgs)) / norm_v
        #print (np.min(imgnew),np.max(imgnew))
        imgnew[imgnew <=0] = 0
        imgnew[imgnew >= 1] = 1
    else:
        imgnew=imgori
    return imgnew


from skimage.exposure import equalize_adapthist


def regular_norm(img):
    #imgori=img.copy()
    img=img.astype(np.float32)

    norm_v=(np.max(img) - np.min(img))
    # (np.min(img))

    imgnew = (img - np.min(img)) / norm_v
    #print (np.min(imgnew),np.max(imgnew))
    imgnew[imgnew <=0] = 0
    imgnew[imgnew >= 1] = 1
    imgnew = equalize_adapthist(imgnew, clip_limit=0.01)

    return imgnew


from scipy.ndimage import binary_dilation
from skimage.measure import regionprops, label
import skimage.io as sio
import glob
import os.path as osp
def roi_correct(result_path,save_path):

        result_files=glob.glob(result_path+'/mask*')
        firstimg=sio.imread(result_files[0])
        roi = np.zeros_like(firstimg) > 0
        files_all = sorted(result_files)
        for fl in files_all:
            roi = roi | (sio.imread(str(fl)) > 0)
        roi = binary_dilation(roi, np.ones(shape=(20, 20)))
        roi = label(roi)
        props = regionprops(roi)
        largest_area, largest_area_id = 0, 0
        for prop in props:
            if prop.area > largest_area:
                largest_area = prop.area
                largest_area_id = prop.label
        roi = (roi == largest_area_id)

        for fl in files_all:
            #print (prediction_instance_id)
            img = sio.imread(str(fl))
            img = img * roi
            print ("!!!!!!!!!!!!!!!!!!!!! start roi")
            if "BF-C2DL-MuSC" in fl:
                img=cv2.dilate(img, kernel=np.ones((3,3)), iterations=1)
            #vis_show=cv2.resize(vis_label(prediction_instance),(0,0),fx=0.5,fy=0.5)
            #cv2.imshow("aa",vis_show)
            #cv2.waitKey()
            sio.imsave(osp.join(save_path,fl.split('/')[-1]), img.astype(np.uint16))