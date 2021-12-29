"""Training code for `solaris` models."""
import os
import numpy as np
from .model_io import get_model, reset_weights
from .datagen_cell_unify import make_data_generator
from .losses import get_loss,get_loss2
from .optimizers import get_optimizer
from .callbacks import get_callbacks
from .torch_callbacks import TorchEarlyStopping, TorchTerminateOnNaN
from .torch_callbacks import TorchModelCheckpoint
from .metrics import get_metrics
import torch
import json
torch.manual_seed(0)
from torch.optim.lr_scheduler import _LRScheduler
import tensorflow as tf
from solaris_rina.nets.zoo.hrnet import init_weights
import os.path as osp
import time
import matplotlib.pyplot as plt

class Trainer(object):
    """Object for training `solaris` models using PyTorch or Keras. """

    def __init__(self, config,dataset_name,branch,GT="ST", custom_model_dict=None, custom_losses=None):
        self.config = config
        self.pretrained = self.config['pretrained']
        self.batch_size = self.config['batch_size']
        self.framework = self.config['nn_framework']
        self.model_name = self.config['model_name']
        self.model_path = self.config.get('model_path', None)
        self.sizecrop=int(self.config['data_specs']['width'])
        self.dataset_name=dataset_name
        self.branch=branch
        save_path=osp.join(self.config['training']['model_dest_path'],self.dataset_name,branch,GT)
        self.save_path=save_path
        #os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)

        try:
            self.num_classes = self.config['data_specs']['num_classes']
        except KeyError:
            self.num_classes = 1
        self.model = get_model(self.model_name, self.framework,
                               self.model_path, self.pretrained,
                               custom_model_dict, self.num_classes)

        #if 'all' in dataset_name:
            #self.train_df, self.val_df = get_train_val_dfs_all(self.config,dataset_name,GT)

        #else:
        self.train_df, self.val_df = get_train_val_dfs(self.config,dataset_name,GT)
        self.train_datagen = make_data_generator(self.config,branch,
                                                 self.train_df, stage='train')
        self.val_datagen = make_data_generator(self.config,branch,
                                               self.val_df, stage='validate')
        self.epochs = self.config['training']['epochs']
        self.optimizer = get_optimizer(self.framework, self.config)
        self.lr = self.config['training']['lr']
        self.custom_losses = custom_losses
        if self.branch=="center":
            self.loss = get_loss2(self.framework,
                             self.config['training'].get('loss'),
                             self.config['training'].get('loss_weights'),
                             self.custom_losses)
        else:
            self.loss = get_loss(self.framework,
                                  self.config['training'].get('loss'),
                                  self.config['training'].get('loss_weights'),
                                  self.custom_losses)
        self.checkpoint_frequency = self.config['training'].get('checkpoint_'
                                                                + 'frequency')
        self.config['training']['callbacks']['model_checkpoint']['filepath']=osp.join(self.config['training']['callbacks']['model_checkpoint']['filepath'],self.dataset_name,branch,GT,"hr_best.pth")
        self.callbacks = get_callbacks(self.framework, self.config)
        self.metrics = get_metrics(self.framework, self.config)
        self.verbose = self.config['training']['verbose']
        if self.framework in ['torch', 'pytorch']:
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.gpu_count = torch.cuda.device_count()
            else:
                self.gpu_count = 0


        self.is_initialized = False
        self.stop = False

        self.initialize_model()
        #print (self.config['pretrained_rina'])
        self.model=init_weights(self.model,self.config['pretrained_rina'],self.model_path)

    def initialize_model(self):
        """Load in and create all model training elements."""
        if not self.pretrained:
            self.model = reset_weights(self.model, self.framework)

        if self.framework == 'keras':
            self.model = self.model.compile(optimizer=self.optimizer,
                                            loss=self.loss,
                                            metrics=self.metrics['train'])

        elif self.framework == 'torch':
            if self.gpu_available:
                self.model = self.model.cuda()
                if self.gpu_count > 1:
                    self.model = torch.nn.DataParallel(self.model)
            # create optimizer
            if self.config['training']['opt_args'] is not None:
                self.optimizer = self.optimizer(
                    self.model.parameters(), lr=self.lr,
                    **self.config['training']['opt_args']
                )
            else:
                self.optimizer = self.optimizer(
                    self.model.parameters(), lr=self.lr
                )
            # wrap in lr_scheduler if one was created
            for cb in self.callbacks:
                if isinstance(cb, _LRScheduler):
                    self.optimizer = cb(
                        self.optimizer,
                        **self.config['training']['callbacks'][
                            'lr_schedule'].get(['schedule_dict'], {})
                        )
                    # drop the LRScheduler callback from the list
                    self.callbacks = [i for i in self.callbacks if i != cb]

        self.is_initialized = True

    def train(self):
        """Run training on the model."""
        if not self.is_initialized:
            self.initialize_model()


        if self.framework == 'torch':
#            tf_sess = tf.Session()

            #path_m = "/local/pal_users/rbync/documents/celltracking/codes/dockerB/wdata/models_pretrain_allrina_distmask/hrnet/hrnet_best_epoch115_0.6050000190734863.pth"
            #model_rina = torch.load(path_m)
            #print('=> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!loading pretrained model {}'.format(path_m))
            #print (model_rina.keys())
            #self.model.load_state_dict(model_rina)
            for epoch in range(self.epochs):
                start_time=time.time()
                if self.verbose:
                    print('Beginning training epoch {}'.format(epoch))
                # TRAINING
                self.model.train()
                for batch_idx, batch in enumerate(self.train_datagen):
                    #print (batch_idx,len(self.train_datagen))
                    if torch.cuda.is_available():
                        if self.config['data_specs'].get('additional_inputs',
                                                         None) is not None:
                            data = []
                            for i in ['image'] + self.config[
                                    'data_specs']['additional_inputs']:
                                data.append(torch.Tensor(batch[i]).cuda())
                        else:
                            data = batch['image'].cuda()
                        target = batch['mask'].cuda().float()
                        mask = batch['dist_mask'].cuda().float()

                    else:
                        if self.config['data_specs'].get('additional_inputs',
                                                         None) is not None:
                            data = []
                            for i in ['image'] + self.config[
                                    'data_specs']['additional_inputs']:
                                data.append(torch.Tensor(batch[i]))
                        else:
                            data = batch['image']
                        target = batch['mask'].float()
                        mask = batch['dist_mask'].float()

                    self.optimizer.zero_grad()
                    output = self.model(data)
                    #print ("outputsize",output.size())
                    #print (output.size(),target.size(),mask.size())
                    if self.branch=="center":
                        loss = self.loss(output, target,mask)
                    else:
                        loss = self.loss(output, target, mask)
                    loss.backward()
                    self.optimizer.step()

                    if self.verbose and batch_idx % 1 == 0:

                        print('    loss at batch {}: {}'.format(
                            batch_idx, loss), flush=True)

                print ("epoch time",time.time()-start_time)
                # VALIDATION
                if epoch%5==0:
                  with torch.no_grad():
                    self.model.eval()
                    torch.cuda.empty_cache()
                    val_loss = []
                    for batch_idx, batch in enumerate(self.val_datagen):
                        if torch.cuda.is_available():
                            if self.config['data_specs'].get(
                                    'additional_inputs', None) is not None:
                                data = []
                                for i in ['image'] + self.config[
                                        'data_specs']['additional_inputs']:
                                    data.append(torch.Tensor(batch[i]).cuda())
                            else:
                                data = batch['image'].cuda()
                            target = batch['mask'].cuda().float()
                            mask=batch['dist_mask'].cuda().float()

                        else:
                            if self.config['data_specs'].get(
                                    'additional_inputs', None) is not None:
                                data = []
                                for i in ['image'] + self.config[
                                        'data_specs']['additional_inputs']:
                                    data.append(torch.Tensor(batch[i]))
                            else:
                                data = batch['image']
                            target = batch['mask'].float()
                            mask=batch['dist_mask'].float()
                        val_output = self.model(data)
                        #see = (val_output.detach().cpu().numpy())[0, 0, :, :]
                        #print (np.max(see))
                        #plt.imshow(np.concatenate([see, batch['mask'][0, 0, :, :], batch['dist_mask'][0, 0, :, :]], 1))
                        #plt.show()
                        if self.branch == "center":
                            loss_a = self.loss(val_output, target,mask)
                        else:
                            loss_a = self.loss(val_output, target,mask)

                        val_loss.append(loss_a)
                    val_loss = torch.mean(torch.stack(val_loss))
                    if self.verbose:
                      print()
                      print('    Validation loss at epoch {}: {}'.format(
                        epoch, val_loss))
                      print()

                    check_continue = self._run_torch_callbacks(
                      loss.detach().cpu().numpy(),
                      val_loss.detach().cpu().numpy())
                    if not check_continue:
                      break

            self.save_model()

    def _run_torch_callbacks(self, loss, val_loss):
        for cb in self.callbacks:
            if isinstance(cb, TorchEarlyStopping):
                cb(val_loss)
                if cb.stop:
                    if self.verbose:
                        print('Early stopping triggered - '
                              'ending training')
                    return False

            elif isinstance(cb, TorchTerminateOnNaN):
                cb(val_loss)
                if cb.stop:
                    if self.verbose:
                        print('Early stopping triggered - '
                              'ending training')
                    return False

            elif isinstance(cb, TorchModelCheckpoint):
                # set minimum num of epochs btwn checkpoints (not periodic)
                # or
                # frequency of model saving (periodic)
                # cb.period = self.checkpoint_frequency

                if cb.monitor == 'loss':
                    cb(self.model, loss_value=loss)
                elif cb.monitor == 'val_loss':
                    cb(self.model, loss_value=val_loss)
                elif cb.monitor == 'periodic':
                    # no loss_value specification needed; defaults to `loss`
                    # cb(self.model, loss_value=loss)
                    cb(self.model)

        return True

    def save_model(self):
        """Save the final model output."""

        outpath=osp.join(self.save_path,"hrnet_final.pth")
        if isinstance(self.model, torch.nn.DataParallel):

                torch.save(self.model.module.state_dict(),outpath
                           )
        else:
                torch.save(self.model.state_dict(),
                           outpath)



def get_sub(datasetlist,da_name):
    subset_list=[]
    for da_item in datasetlist:
        if da_name in da_item:
            subset_list.append(da_item)

    return subset_list

dataset3Dlist=['Fluo-C3DH-A549','Fluo-C3DL-MDA231','Fluo-C3DH-H157','Fluo-N3DH-CE','Fluo-N3DH-CHO']
dataset2Dlist=['BF-C2DL-HSC',
               'BF-C2DL-MuSC',
               'DIC-C2DH-HeLa',
               'Fluo-C2DL-MSC',
               'Fluo-N2DH-GOWT1',
               'Fluo-N2DL-HeLa',
               'PhC-C2DH-U373',
               'PhC-C2DL-PSC']
dataset2DlistnoBF=[
               'DIC-C2DH-HeLa',
               'Fluo-C2DL-MSC',
               'Fluo-N2DH-GOWT1',
               'Fluo-N2DL-HeLa',
               'PhC-C2DH-U373',
               'PhC-C2DL-PSC']

def get_train_val_dfs(config,dataset,gttype):
    pathload = osp.join(config['training_data_names'], "ids_all" + gttype + ".npy")

    data_all=np.load(pathload,allow_pickle=True).item()


    if dataset =="allBF":
        dalist=['BF-C2DL-HSC',
               'BF-C2DL-MuSC']
        sub_train=[]
        sub_val=[]
        for da in dalist:
            sub_train_da = get_sub(data_all['train'], da)
            sub_val_da = get_sub(data_all['val'], da)

            sub_train.extend(sub_train_da)
            sub_val.extend(sub_val_da)


    elif dataset == "3Dtrain":
        dalist = ['Fluo-C3DH-A549', 'Fluo-C3DL-MDA231', 'Fluo-C3DH-H157', 'Fluo-N3DH-CE', 'Fluo-N3DH-CHO']

    elif dataset=="all":
        dalist=[]
        dalist.extend(dataset3Dlist)
        dalist.extend(dataset2DlistnoBF)

    else:
        dalist=[dataset]

    sub_train = []
    sub_val = []

    for da in dalist:
        sub_train_da = get_sub(data_all['train'], da)
        sub_val_da = get_sub(data_all['val'], da)

        sub_train.extend(sub_train_da)
        sub_val.extend(sub_val_da)

    return sub_train,sub_val


def get_train_val_dfs_all(config,dataset,gttype):
    train_df = np.load(config["training_data_names"])
    val_df = np.load(config["validation_data_names"])
    return train_df, val_df
