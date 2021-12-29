import os
import torch
import numpy as np
from model_io import get_model
import matplotlib.pyplot as plt
import skimage.io as sio
import cv2
from skimage.exposure import equalize_adapthist


da_scale = {'Fluo-C2DL-MSC': 0.35,
            'Fluo-C3DH-H157': 0.35,
            'Fluo-C3DL-MDA231': 2,
            'Fluo-N3DH-CE': 0.5,
            'Fluo-N3DH-CHO': 0.6,
            'PhC-C2DL-PSC': 3,
            'BF-C2DL-MuSC':0.75,
            'BF-C2DL-HSC':0.75,
            }


def convert_rgb(gray):
    imgshow = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
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



def normalize(read_mask):

    read_mask=(read_mask-np.min(read_mask))/(np.max(read_mask)-np.min(read_mask))
    return read_mask
import torchvision.transforms as transforms
from scipy.stats import stats

class Inferer_singleimg(object):

    def __init__(self,model_path):
        self.batch_size = 1

        model = get_model(model_path)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.sigmoid=torch.nn.Sigmoid()
        #if not os.path.isdir(self.output_dir):
        #    os.makedirs(self.output_dir)
        with torch.no_grad():
            model.eval()
        self.model = model.cuda()

    def __call__(self, im_path,contrast=False):
        """Run inference.
        Arguments
        ---------
        infer_df : :class:`pandas.DataFrame` or `str`
            A :class:`pandas.DataFrame` with a column, ``'image'``, specifying
            paths to images for inference. Alternatively, `infer_df` can be a
            path to a CSV file containing the same information.  Defaults to
            ``None``, in which case the file path specified in the Inferer's
            configuration dict is used.
        """



        def inf_img(im,transforms):

            imori_size=im.shape
            da=im_path[-1]
            if da in da_scale.keys():
                scale=da_scale[da]
            else:
                scale=1.
            im=contrast_strech(im)
            im = 255 * equalize_adapthist(im, clip_limit=0.01)
            #print ("mask image range",np.min(im),np.max(im))

            im = np.asarray(im, dtype=np.uint8)
            im = convert_rgb(im)
            im = cv2.resize(im, (0,0),fx=scale,fy=scale)
            #plt.imshow(im)
            #plt.show()
            im = transforms(im)
            #print (im.shape)
            #print ("mask image range",np.min(im.numpy()),np.max(im.numpy()))
            output_arr = np.expand_dims(im, 0).astype(np.float32)
            #print (output_arr.shape)
            #output_arr = np.moveaxis(output_arr, 3, 1)
            return output_arr,imori_size,scale

        im_all= sio.imread(im_path[0])
        #im_all = contrast_strech(im_all)

        pred_return=np.zeros_like(im_all,dtype=np.float32)
        for tt in range(im_all.shape[0]):
          im=im_all[tt,:,:]
          inf_input,imori_size,scale = inf_img(im,self.transforms)

          if scale<3:
            inf_input = torch.from_numpy(inf_input).float().cuda()
            #print ("input size",inf_input.size)
            with torch.no_grad():
                #print (inf_input.size())
                subarr_preds = self.sigmoid(self.model(inf_input))
                subarr_preds = subarr_preds.cpu().data.numpy()
          else:
            num_split = 2

            shape_n = [int(inf_input.shape[2] / num_split), int(inf_input.shape[3] / num_split)]
            #print (shape_n)
            subarr_preds = np.zeros_like(inf_input)

            inf_input = torch.from_numpy(inf_input).float().cuda()
            # print ("input size",inf_input.size)
            with torch.no_grad():
                # print (inf_input.size())
                for m in range(num_split):
                    for n in range(num_split):
                        subtest = inf_input[:, :, shape_n[0] * m:shape_n[0] * (m + 1),
                                  shape_n[1] * n:shape_n[1] * (n + 1)]

                        pred_l = self.sigmoid(self.model(subtest))
                        pred_l = pred_l.cpu().data.numpy()
                        subarr_preds[:, :, shape_n[0] * m:shape_n[0] * (m + 1),
                        shape_n[1] * n:shape_n[1] * (n + 1)] = pred_l


          pred=subarr_preds[0,0,:,:]
          #plt.imshow(pred)
          #plt.show()
          pred = cv2.resize(pred, (imori_size[1],imori_size[0]))
          pred_return[tt,:,:]=pred

        return pred_return





