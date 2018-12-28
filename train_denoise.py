from __future__ import division
import os 
import pprint
import sys
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from convsparse_net import LISTAConvDictADMM
import common
from common import save_train, load_train, clean
from common import gaussian, normilize, nhwc_to_nchw
from common import reconsturction_loss, init_model_dir
from test_denoise import plot_res
from datasets import  DatasetFromNPZ
import arguments
import test_denoise

USE_CUDA = torch.cuda.is_available()
FILE_PATH = os.path.dirname(os.path.abspath(__file__))

def _pprint(stuff):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(stuff)

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


def step(model, img, img_n, optimizer, criterion):
    """ Train step !!"""
    optimizer.zero_grad()

    output, _sc_n = model(img_n)
#    output, sc_tar = model(img)

    results = [
        output[..., 3:-3, 3:-3],
    ]

    targets = [
        img[..., 3:-3, 3:-3],
    ]

    loss = criterion(results, targets)

    loss.backward()
    optimizer.step()

    return float(loss), output.cpu()

def maybe_save_model(
        model,
        opt,
        schd,
        epoch,
        save_path,
        curr_val,
        other_values,
        model_path=None):
    path = model_path if model_path is not None else ''

    if not other_values or curr_val > max(other_values):
        path = save_train(save_path, model, opt, schd, epoch)
        print('saving model at path %s'%path)
        clean(save_path, save_count=10)

    return path

def run_valid(model, data_loader, criterion, logdir, name, should0_plot=False):
    """
    Run over whole valid set calculate psnr and critirion loss.
    """
    loss = 0
    psnr = 0

    for img, img_n in data_loader:
        _out, _ = model(img_n)
        loss += float(criterion(img.data, _out.data))
        psnr += common.psnr(img.data.cpu().numpy(), _out.data.cpu().numpy())

    img, img_n = data_loader.dataset[0]
    output, _ = model(img_n.unsqueeze(0))
    plot_res(
        img.data.cpu().numpy(),
        img_n.data.cpu().numpy(),
        output.data.cpu().numpy(),
        name,
        logdir
    )
    return loss / len(data_loader), psnr / len(data_loader)

def train(model, args):

    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    train_loader, valid_loader = get_train_valid_loaders(args['dataset_path'], args['batch_size'], args['noise'])

    #TODO(hillel): this is ugly fix it!! simple l2 dist for valid more complex for train
    valid_loss = reconsturction_loss(use_cuda=True)


    criterion = common.get_criterion(
        losses_types=['l1'],
        factors=[1.0],
        use_cuda=USE_CUDA
    )

    print('train args:')
    _pprint(args)

    if args['load_path'] != '':
        ld_p = args['load_path']
        print('loading from %s'%ld_p)
        load_train(ld_p, model, optimizer, scheduler)
        print('Done!')

    model_path = None
    _train_loss = []
    _valid_loss = []
    _valid_psnr = []
    running_loss = 0
    valid_every = int(0.05 * len(train_loader))

    itr = 0
    for e in range(args['epoch']):
        print('Epoch number {}'.format(e))
        for img, img_n in train_loader:
            itr += 1

            _loss, _ = step(model, img, img_n, optimizer, criterion=criterion)
            running_loss += float(_loss)

            if itr % valid_every == 0:
                _train_loss.append(running_loss / valid_every)
                _v_loss, _v_psnr = run_valid(
                    model,
                    valid_loader,
                    valid_loss,
                    args['save_dir'],
                    f'perf_iter{itr}', 
                    itr == valid_every
                )

                scheduler.step(_v_loss)

                model_path = maybe_save_model(
                    model,
                    optimizer,
                    scheduler,
                    e,
                    args['save_dir'],
                    _v_psnr,
                    _valid_psnr,
                    model_path
                )
                _valid_loss.append(_v_loss)
                _valid_psnr.append(_v_psnr)
                print("epoch {} train loss: {} valid loss: {}, valid psnr: {}".format(e,
                    running_loss / valid_every, _v_loss, _v_psnr))
                running_loss = 0
    return model_path, _valid_loss[-1], _valid_psnr[-1]

def build_model(args):

    print("model args")
    _pprint(args)

    model = LISTAConvDictADMM(
        num_input_channels=args['num_input_channels'],
        num_output_channels=args['num_output_channels'],
        kc=args['kc'], 
        ks=args['ks'],
        ista_iters=args['ista_iters'],
        iter_weight_share=args['iter_weight_share'],
        use_sigmoid=args.get('use_sigmoid', False)
    )
    if USE_CUDA:
        model = model.cuda()
    return model

def main(args_file):
    args = arguments.load_args(args_file)
    log_dir  = init_model_dir(args['train_args']['log_dir'], args['train_args']['name'])
    arguments.logdictargs(os.path.join(log_dir, 'params.json'), args)
    args['train_args']['save_dir'] = log_dir
    args['train_args']['log_dir'] = log_dir
    model = build_model(args['model_args'])
    model_path, valid_loss, valid_psnr = train(model, args['train_args'])

    args['test_args']['load_path'] = model_path
    args['train_args']['final_loss'] = valid_loss
    args['train_args']['final_psnr'] = valid_psnr
    arguments.logdictargs(os.path.join(log_dir, 'params.json'), args)

    if args['test_args']['testset_path']:
        psnrs, res = test_denoise.test(args['model_args'],
             model_path,
             args['train_args']['noise'], 
             args['test_args']['testset_path']
             )
        args['test_args']['final_psnrs'] = psnrs
        arguments.logdictargs(os.path.join(log_dir, 'params.json'), args)
        for idx, ims in enumerate(res):
            test_denoise.plot_res(ims[0], ims[1], ims[2], idx, args['train_args']['log_dir'])
    else:
        print('no test path provided skipping test run')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_file', default='')
    arg_file = parser.parse_args().args_file

    main(arg_file)
