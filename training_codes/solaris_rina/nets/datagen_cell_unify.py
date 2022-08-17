import numpy as np
from torch.utils.data import Dataset, DataLoader
#from .transform import _check_augs, process_aug_dict, scale
#from ..utils.core import _check_df_load
#from ..utils.io import imread, _check_channel_order
from skimage.exposure import equalize_adapthist
import json
import os.path as osp
import skimage.io as sio



def _check_channel_order(im_arr):
    im_shape = im_arr.shape
    if len(im_shape) == 3:  # doesn't matter for 1-channel images
        if im_shape[0] > im_shape[2]:
            # in [Y, X, C], needs to be in [C, Y, X]
            im_arr = np.moveaxis(im_arr, 2, 0)
    elif len(im_shape) == 4:  # for a whole minibatch
        if im_shape[1] > im_shape[3]:
            # in [Y, X, C], needs to be in [C, Y, X]
            im_arr = np.moveaxis(im_arr, 3, 1)

    return im_arr




def make_data_generator(config, branch,df, stage='train'):

    try:
        num_classes = config['data_specs']['num_classes']
    except KeyError:
        num_classes = 1
    sizecrop=int(config['data_specs']['width'])


    dataset = TorchDataset(branch,
            df,
            batch_size=config['batch_size'],
            sizecrop=sizecrop,
            num_classes=num_classes)

        # set up workers for DataLoader for pytorch
    data_workers = config['data_specs'].get('data_workers')
    if data_workers == 1 or data_workers is None:
            data_workers = 0  # for DataLoader to run in main process
    data_gen = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=config['training_augmentation']['shuffle'],
            num_workers=data_workers)

    return data_gen


from scipy import ndimage


import random
import cv2


def normalize(read_mask):

    read_mask=(read_mask-np.min(read_mask))/(np.max(read_mask)-np.min(read_mask))
    return read_mask


def generate_distmask(read_mask,maxd=20):
    #plt.imshow(read_mask)
    #plt.show()
    read_mask=np.asarray(read_mask,dtype=np.uint8)
    kernel = np.ones((5, 5), np.uint8)

    dia_mask = cv2.dilate(read_mask, kernel, iterations=1)
    cc=dia_mask-read_mask
    #plt.imshow(cc)
    #plt.show()
    distance_map = ndimage.distance_transform_edt(1-cc)
    #plt.imshow(distance_map)
    #plt.show()
    distance_map=(distance_map)
    distance_map[distance_map > maxd]=maxd
    distance_map=maxd-distance_map

    distance_map=normalize(distance_map)

    return distance_map

def generate_shapemarker_each(read_mask,kernel_da,min_da):

    dia_mask=np.zeros_like(read_mask)
    for ids in np.unique(read_mask):
        if ids>0:
            area=np.sum(read_mask==ids)
            kernel_size=max(min(int(area*0.015),kernel_da),min_da)
            #print (area,kernel_size)

            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            ids_mask=np.asarray(255*(read_mask==ids),dtype=np.uint8)

            dia_mask = dia_mask+cv2.erode(ids_mask, kernel, iterations=1)


    return dia_mask

def get_centermask(da_name,mask):
    if da_name not in ['BF-C2DL-HSC' ,'BF-C2DL-MuSC','Fluo-C2DL-MSC', 'Fluo-N2DH-GOWT1', 'Fluo-N2DL-HeLa', 'Fluo-C3DL-MDA231', 'Fluo-N3DH-CHO',
                       'Fluo-C3DH-H157']:
        if da_name in ['PhC-C2DL-PSC']:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        else:

            kernel = np.ones((5, 5), np.uint8)

            mask = cv2.dilate(mask, kernel, iterations=2)
    else:
        mask=mask

    return mask

from scipy.stats import stats

def convert_rgb(gray):
    imgshow = np.zeros((gray.shape[0], gray.shape[1], 3),dtype=np.uint8)
    imgshow[:, :, 0] = gray
    imgshow[:, :, 1] = gray
    imgshow[:, :, 2] = gray
    return imgshow


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

import torchvision.transforms as transforms




thre = {"Fluo-N2DL-HeLa": [5, 3], "DIC-C2DH-HeLa": [20, 3], "Fluo-C2DL-MSC": [5, 3], "PhC-C2DH-U373": [10, 3],
            "PhC-C2DL-PSC": [3, 3], 'Fluo-N2DH-GOWT1': [10, 3], 'BF-C2DL-HSC': [5, 5], 'BF-C2DL-MuSC': [7, 3],
        'Fluo-C3DH-A549':[5,3],
        'Fluo-C3DL-MDA231':[5,3],
        'Fluo-C3DH-H157':[5,3],
       'Fluo-N3DH-CE':[5,3],
       'Fluo-N3DH-CHO':[10,3],}

da_scale = {'Fluo-C2DL-MSC': 0.35,
            'Fluo-C3DH-H157': 0.35,
            'Fluo-C3DL-MDA231': 2,
            'Fluo-N3DH-CE': 0.5,
            'Fluo-N3DH-CHO': 0.6,
            'PhC-C2DL-PSC': 3,
            'BF-C2DL-MuSC':0.75,
            'BF-C2DL-HSC':0.75,
            }

import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def rand_crop(image, label,dist,cropsize):
    h, w = image.shape[:-1]

    asv=int(cropsize/2.)
    x = random.randint(asv, w - asv)
    y = random.randint(asv, h - asv)
    image = image[y-asv:y + asv, x-asv:x + asv]
    label = label[y-asv:y + asv, x-asv:x +asv]
    dist = dist[y-asv:y + asv, x-asv:x +asv]

    return image, label,dist


dataset3Dlist=['Fluo-C3DH-A549','Fluo-C3DL-MDA231','Fluo-C3DH-H157','Fluo-N3DH-CE','Fluo-N3DH-CHO']
import random
import glob
class TorchDataset(Dataset):


    def __init__(self, branch,df, batch_size,sizecrop,stage="train",num_classes=1):

        super().__init__()

        self.branch=branch
        self.df = df
        self.cropsize=sizecrop
        self.stage=stage
        self.aug_el=iaa.Sequential([iaa.Affine(rotate=(-45, 45)),iaa.Flipud(0.5),iaa.Fliplr(0.5),iaa.PiecewiseAffine(scale=(0.01, 0.05)),iaa.Sometimes(0.5,iaa.ElasticTransformation(alpha=(25.0,75.0),sigma=10.0)),iaa.Sometimes(0.3,iaa.blur.GaussianBlur(sigma=[1.0,2.75])), iaa.Sometimes(0.5,iaa.GammaContrast((0.5, 2.0)))])


        self.batch_size = batch_size
        self.n_batches = int(np.floor(len(self.df)/self.batch_size))
        self.num_classes = num_classes
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get one image, mask pair"""
        # Generate indexes of the batch
        frame=self.df[idx]
        #print (frame)
        branch=self.branch
        #print (idx,frame)
        gttype=frame.split('_')[1]
        da_name = frame.split('_')[0]

        data_dir = osp.join("../Data/", "train", da_name)

        if da_name in dataset3Dlist:
            seq_name = frame.split('_')[2]
            imgname = frame.split('_')[3]

            img_path = osp.join(data_dir, seq_name, "t" + imgname + ".tif")

            image = sio.imread(img_path)
            #print (img_path)
            if gttype=="GT":
              if da_name not in ['Fluo-C3DH-A549']:
                mask_listcheckpath=osp.join(data_dir, seq_name+"_"+gttype, "SEG", "man_seg_" + imgname+"_*.tif")
                #print (mask_listcheckpath)
                slicelist= glob.glob(mask_listcheckpath)
                stn=random.randint(0,len(slicelist)-1)
                slice_name =slicelist[stn].split('/')[-1].split('.tif')[0].split('_')[-1]
                #print ("hhhhhh",slice_name)
              else:
                slice_name=frame.split('_')[4]
            else:
                #print (frame)
                slice_name=frame.split('_')[4]

            #print ("slicename",slice_name)

            image=image[int(slice_name),:,:]


        else:
            seq_name = frame.split('_')[2]
            imgname = frame.split('_')[3]
            img_path = osp.join(data_dir, seq_name, "t" + imgname + ".tif")
            image = sio.imread(img_path)
            #slice_name=0


        #print ("image shape",image.shape)
        image=contrast_strech(image)
        image = 255*equalize_adapthist(image, clip_limit=0.01)

        if branch == "center":

            center_path = osp.join(data_dir, seq_name + "_GT" , "TRA", "man_track" + imgname + ".tif")
            mask = sio.imread(center_path)
            if da_name in dataset3Dlist:
                mask=mask[int(slice_name),:,:]

            mask = get_centermask(da_name, mask)
            mask = 1.0 * (mask > 0)

        else:
            if gttype=="GT":
                if da_name in dataset3Dlist:
                    if da_name not in ['Fluo-C3DH-A549']:

                        mask_path = osp.join(data_dir, seq_name + "_" + gttype, "SEG", "man_seg_" + imgname +"_"+(slice_name)+".tif")

                        mask = sio.imread(mask_path)

                    else:
                        mask_path = osp.join(data_dir, seq_name + "_" + gttype, "SEG", "man_seg" + imgname +".tif")
                        mask = sio.imread(mask_path)[int(slice_name),:,:]



                else:
                    mask_path = osp.join(data_dir, seq_name + "_" + gttype, "SEG", "man_seg" + imgname +".tif")

                    mask = sio.imread(mask_path)

            else:
                mask_path = osp.join(data_dir, seq_name + "_" + gttype, "SEG", "man_seg" + imgname + ".tif")
                #print ("maskpath",mask_path)
                mask = sio.imread(mask_path)
                if da_name in dataset3Dlist:
                    mask=mask[int(slice_name),:,:]
            if branch=="shapemarker":
                mask = generate_shapemarker_each(mask, thre[da_name][0], thre[da_name][1])


        scale=1

        if da_name in da_scale.keys():
            scale=da_scale[da_name]
            image=cv2.resize(image,(0,0),fx=scale,fy=scale)
            mask=cv2.resize(mask,(0,0),fx=scale,fy=scale)

        else:
            if (da_name not in ['Fluo-C3DH-A549']):
                random_scale = 0.1 * random.randint(8, 12)
                image = cv2.resize(image, (0, 0), fx=random_scale, fy=random_scale)
                mask = cv2.resize(mask, (0, 0), fx=random_scale, fy=random_scale)

        if scale>=1 and (da_name not in ['Fluo-C3DH-A549','BF-C2DL-HSC','BF-C2DL-MuSC']):
            c_y=image.shape[0]
            c_x=image.shape[1]

            y_use_list=list(range(150,c_y-150,128))
            y_use_list.append(c_y-150)

            x_use_list = list(range(150, c_x - 150, 128))
            x_use_list.append(c_x - 150)

            random.shuffle(y_use_list)
            random.shuffle(x_use_list)

            y_use=y_use_list[0]
            x_use=x_use_list[0]

            image=image[y_use-150:y_use+150,x_use-150:x_use+150]
            mask=mask[y_use-150:y_use+150,x_use-150:x_use+150]

        #if len(mask.shape) == 2:
        #    mask = mask[:, :, np.newaxis]
        #plt.imshow(mask)
        #plt.show()
        mask=1.0*(mask>0)


        mask = np.asarray(np.expand_dims(mask, -1) , dtype=np.int32)
        #print ("mask value",np.max(mask),np.min(mask))


        image=np.asarray(image,dtype=np.uint8)
        if self.stage=="train":
        #print (self.stage)
            segmap = SegmentationMapsOnImage(mask, shape=image.shape)
            image, mask_new = self.aug_el(image=image, segmentation_maps=segmap)
            mask_new=mask_new.get_arr()

        #mask_new=mask
        #print ("maskshape",mask_new.shape,type(mask_new))
        image = convert_rgb(image)

        #print (type(mask))
        dist_mask=generate_distmask(mask_new[:,:,0])
        #dist_mask = dist_mask[:, :, np.newaxis]
        #print ("!!!!!!!!!!!!!",np.max(dist_mask),np.min(dist_mask))
        #print ("maskshapecheck",mask_new.shape,)

        #print (image.shape, mask_new.shape, dist_mask.shape)
        image,mask_new,dist_mask=rand_crop(image, mask_new, dist_mask, self.cropsize)
        #print (image.shape,mask_new.shape)
        #plt.imshow(np.concatenate([image[:,:,0],255*mask_new[:,:,0],255*dist_mask,127*mask_new[:,:,0]+127*dist_mask],1))
        #plt.imshow(dist_mask)
        #plt.show()
        dist_mask = dist_mask[:, :, np.newaxis]
        sample = {'image': image, 'mask': mask_new, 'dist_mask':dist_mask}

        #print ("image",np.min(sample['image']),np.max(sample['image']))
        #print ("mask",np.min(sample['mask']),np.max(sample['mask']))
        #print ("dist",np.min(sample['dist_mask']),np.max(sample['dist_mask']))

        sample['image'] = self.transforms(sample['image'])
        #print (np.min(sample['image'].numpy()),np.max(sample['image'].numpy()))
        sample['image'] = _check_channel_order(sample['image'])
        sample['mask'] = _check_channel_order(sample['mask']).astype(np.float32)

        sample['dist_mask']=_check_channel_order(dist_mask,).astype(np.float32)

        return sample


if __name__ == "__main__":


    da="all"
    gttype="GT+ST"
    if da=="all":
        pathload = "/home/rbync/data/top1splits/splits/ids_" + da + gttype + ".json"
        with open(pathload) as f:
            data_t = json.load(f)
            if gttype == "GT+ST":
                train_dataset=[]
                val_dataset=[]
                train_dataset_d = data_t['train']
                val_dataset_d = data_t['val']
                #print (train_dataset_d)
                for aa in train_dataset_d:
                    #print (aa)
                    if 'BF' not in aa:
                        #print (aa)

                        train_dataset.append(aa)
                for bb in val_dataset_d:
                     if 'BF' not in bb:
                         #print (aa)

                         val_dataset.append(bb)


            else:
                train_dataset_d = data_t['train']
                val_dataset_d = data_t['val']
                train_dataset=[]
                val_dataset=[]
                for kys in train_dataset_d.keys():
                    if 'BF' not in kys:
                        print (kys)

                        train_dataset.extend(train_dataset_d[kys])
                        val_dataset.extend(val_dataset_d[kys])
            print (train_dataset)
            dataset = TorchDataset("shapemarker",
                     train_dataset,
                     batch_size=1,
                     sizecrop=256,
                     num_classes=1,)

            for i in range(len(dataset.df)):
               sample=dataset[i]

               print ("check output",sample['image'].shape,sample['mask'].shape,sample['dist_mask'].shape)
               print ("check range",np.max(sample['image'].numpy()))
               print (np.max(sample['mask']))
               print (np.max(sample['dist_mask']))