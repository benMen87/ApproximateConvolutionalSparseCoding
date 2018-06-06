from __future__ import division
import os 
import sys
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from convsparse_net import LISTAConvDictADMM
from common import save_train, load_train, clean
from common import gaussian, normilize, nhwc_to_nchw
from common import reconsturction_loss, init_model_dir
from datasets import  DatasetFromNPZ
from torch.utils.data import DataLoader
import arguments

USE_CUDA = torch.cuda.is_available()
FILE_PATH = os.path.dirname(os.path.abspath(__file__))

def get_train_valid_loaders(dataset_path, batch_size, noise):
   
    def pre_process_fn(_x): return normilize(nhwc_to_nchw(_x), 255) 
    def input_process_fn(_x): return gaussian(_x, is_training=True, mean=0, stddev=normilize(noise, 255))

    train_loader = DatasetFromNPZ(npz_path=dataset_path,
                              key='TRAIN', use_cuda=USE_CUDA,
                              pre_transform=pre_process_fn,
                              inputs_transform=input_process_fn)
    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)

    valid_loader = DatasetFromNPZ(npz_path=dataset_path,
                              key='VAL', use_cuda=USE_CUDA,
                              pre_transform=pre_process_fn,
                              inputs_transform=input_process_fn)
    valid_loader = DataLoader(valid_loader, batch_size=1, shuffle=True)
    return train_loader, valid_loader

def step(model, img, img_n, optimizer=None, criterion=None):
    if optimizer is not None: 
       optimizer.zero_grad()
    output = model(img_n)
    if criterion is not None:
        loss = criterion(output[..., 3:-3, 3:-3], img[..., 3:-3, 3:-3])
        if optimizer:
            loss.backward()
            optimizer.step()
        return loss, output.cpu()
    return float(output)

def maybe_save_model(model, opt, schd, epoch, save_path, curr_val, other_values):
    
    def no_other_values(other_values):
        return len(other_values) == 0
    if no_other_values(other_values) or curr_val < min(other_values):
        print('saving model...')
        save_train(save_path, model, opt, schd, epoch)
        clean(save_path, save_count=10)

def run_valid(model, data_loader, criterion, logdir):
    loss = 0
    for img, img_n in data_loader:
        _loss, _ = step(model, img, img_n, criterion=criterion)
        loss += float(_loss)

    _, output = step(model, img, img_n, criterion=criterion)
    np.savez(os.path.join(logdir, 'images'), IN=img.data.cpu().numpy(),
        OUT=output.data.cpu().numpy(), NOISE=img_n.data.cpu().numpy())
    return loss / len(data_loader)

def train(model, args):
    
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    recon_loss = reconsturction_loss(factor=0.2, use_cuda=True)
    train_loader, valid_loader = get_train_valid_loaders(args['dataset_path'], args['batch_size'], args['noise'])
    print(args)
    if args['load_path'] != '':
        ld_p = args['load_path']
        print('loading from %s'%ld_p)
        load_train(ld_p, model, optimizer, scheduler)        
        print('Done!')
        
    _train_loss = []
    _valid_loss = []
    running_loss = 0
    valid_every = int(0.1 * len(train_loader))

    itr = 0
    for e in range(args['epoch']):
        print('Epoch number {}'.format(e))
        for img, img_n in train_loader:
            itr += 1

            _loss, _ = step(model, img, img_n, optimizer, recon_loss)
            running_loss += float(_loss)

            if itr % valid_every == 0:
                _train_loss.append(running_loss / valid_every)
                _v_loss = run_valid(model, valid_loader,
                        recon_loss,args['save_dir'])
                scheduler.step(_v_loss)
                maybe_save_model(model, optimizer,
                        scheduler, e, args['save_dir'],
                        _v_loss, _valid_loss)
                _valid_loss.append(_v_loss)
                print("epoch {} train loss: {} valid loss: {}".format(e,
                    running_loss / valid_every, _v_loss))
                running_loss = 0

def build_model(args):
    model = LISTAConvDictADMM(
        num_input_channels=args['num_input_channels'],
        num_output_channels=args['num_output_channels'],
        kc=args['kc'], 
        ks=args['ks'],
        ista_iters=args['ista_iters'],
        iter_weight_share=args['iter_weight_share'],
    )
    if USE_CUDA:
        model = model.cuda()
    return model

def main(args_file):
    args = arguments.load_args(args_file)
    log_dir, save_dir = init_model_dir(args['train_args']['log_dir'], 'trainSess')
    arguments.logdictargs(os.path.join(log_dir, 'params.txt'), args)
    args['train_args']['save_dir'] = save_dir
    args['train_args']['log_dir'] = log_dir
    model = build_model(args['model_args'])
    train(model, args['train_args'])

if __name__ == '__main__':
<<<<<<< 276d6c926762a0105f2dce69396ab672c8fbe17b
    args = {
       'train_args':
        {
            'noise': 20,
            'epoch': 100,
            'batch_size': 5,
            'learning_rate': 1e-3,
            'dataset_path': '/data/hillel/data_sets/pascal320_notst.npz',
            'save_dir': os.path.join(FILE_PATH, 'saved_models'),
        },
        'model_args':
        {
            'num_input_channels': 1,
            'num_output_channels': 1,
            'kc': 64, 
            'ks': 7,
            'ista_iters': 3,
            'iter_weight_share': True,
        }
    }
    main(args)
=======
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg_file', default='')
    arg_file = parser.parse_args().arg_file

    main(arg_file)
>>>>>>> reload and save model
