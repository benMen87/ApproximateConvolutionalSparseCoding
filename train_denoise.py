from __future__ import division
import numpy as np
import torch
from torch import nn
from torch import optim
from convsparse_net import LISTAConvDictADMM
from common import gaussian, normilize, nhwc_to_nchw
from common import reconsturction_loss
from datasets import  DatasetFromNPZ
from torch.utils.data import DataLoader

USE_CUDE = torch.cuda.is_available()

noise = 20
epoch = 10
batch_size = 5
learninig_rate = 1e-3

dataset_path = '/data/hillel/data_sets/pascal320.npz'

def pre_process_fn(_x): return normilize(nhwc_to_nchw(_x), 255) 
def input_process_fn(_x): return gaussian(_x, is_training=True, mean=0, stddev=normilize(noise, 255))


train_loader = DatasetFromNPZ(npz_path=dataset_path,
                              key='TRAIN', use_cuda=USE_CUDE,
                              pre_transform=pre_process_fn,
                              inputs_transform=input_process_fn)
train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)

valid_loader = DatasetFromNPZ(npz_path=dataset_path,
                              key='VAL', use_cuda=USE_CUDE,
                              pre_transform=pre_process_fn,
                              inputs_transform=input_process_fn)
valid_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)

model = LISTAConvDictADMM(
    num_input_channels=1,
    num_output_channels=1,
    kc=64, 
    ks=7,
    ista_iters=3,
    iter_wieght_share=True,
)
if USE_CUDE:
    model = model.cuda()

opt = optim.Adam(model.parameters(), lr=learninig_rate)
recon_loss = reconsturction_loss(factor=0.2, use_cuda=True)

_train_loss = []
_valid_loss = []
running_loss = 0
print_every = 100

itr = 0
for e in range(epoch):
    print('Epoch number {}'.format(e))
    for ims, ims_n in train_loader:
        itr += 1

        opt.zero_grad()
        output = model(ims_n)
        _loss = recon_loss(output[..., 3:-3, 3:-3], ims[..., 3:-3, 3:-3])
        _loss.backward()
        opt.step()

        running_loss += _loss
        if itr % print_every == 0:
            print("epoch {} loss: {}".format(e, running_loss / print_every))
            _train_loss.append(running_loss / print_every)
            running_loss = 0
            np.savez('images', IN=ims.data.cpu().numpy(),
                    OUT=output.data.cpu().numpy())
    
    torch.save(model.state_dict(), './saved_models/e_%d'%e)

