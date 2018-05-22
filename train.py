import numpy as np
import torch.utils.data.DataLoader as data_loader
import torch
from torch import nn
from torch import optim
from convsparse_net import LISTAConvDictADMM
from common import gaussian as get_noise
from common import reconsturction_loss


USE_CUDE = torch.cuda.is_available()

noise = 20
epoch = 1e7
batch_size = 16
learninig_rate = 1e-3

dataset_path = '/data/hillel/data_sets/pascal320.npz'
dataset = np.load(dataset_path)

train_loader = data_loader(dataset=dataset['TRAIN'], batch_size=batch_size, shuffle=True)
val_loader = data_loader(dataset=dataset['VAL'], batch_size=batch_size, shuffle=True)


model = LISTAConvDictADMM(
    num_input_channels=1,
    num_input_channels=1,
    kc=64, 
    ks=7,
    ista_iters=3,
    iter_wieght_share=True,
)
if USE_CUDE:
    model = model.cuda()

opt = optim.Adam(model.parameters(), lr=learninig_rate)
recon_loss = reconsturction_loss(factor=0.2, use_cuda=True)

for e in range(epoch):
    print('Epoch number {}'.format(e))
    for inpt in train_loader:
        input_n = get_noise(inpt, is_training=True, mean=0, stddev=noise)
