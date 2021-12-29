import cv2
import numpy as np
import torch
from skimage import measure, morphology, segmentation
from skimage.morphology import extrema, h_maxima, reconstruction, local_maxima, thin
from scipy import ndimage





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
    labels, nb = ndimage.label(unet_output > 0)
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


def postprocess_mask_and_markers(mask, markers, area_thresh=50):
    # Label markers if not labeled
    markers = measure.label(markers > 0.5)
    # Remove small components
    mask = remove_small_components(mask, area_thresh=area_thresh)

    # Calculate distance transform for watershed
    _, distance = morphology.medial_axis(mask > 0, return_distance=True)

    # Use watershed
    labels = segmentation.watershed(-distance, markers, mask=mask)

    # remove small components
    for i in range(1, labels.max() + 1):
        if np.sum(labels == i) < area_thresh:
            labels[labels == i] = 0

    return labels
