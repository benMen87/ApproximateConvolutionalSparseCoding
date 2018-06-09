import torch
import common
from common import gaussian, normilize, nhwc_to_nchw, to_np
import numpy as np
from datasets import DatasetFromFolder
from torch.utils.data import DataLoader
from convsparse_net import LISTAConvDictADMM
import arguments
import matplotlib.pyplot as plt

DEFAULT_IMG_PATH = '/data/hillel/data_sets/test_images/'
USE_CUDA = torch.cuda.is_available()


def plot_res(img, img_n, res, name):
    
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('clean')
    plt.subplot(132)
    plt.imshow(img_n, cmap='gray')
    plt.title('noise psnr {:.2f}'.format(common.psnr(img, img_n)))
    plt.subplot(133)
    plt.imshow(res, cmap='gray')
    plt.title('noise psnr {:.2f}'.format(common.psnr(img, res)))
    plt.savefig('tmp_log/res_{}'.format(name))
    plt.clf()
   

def test(args, saved_model_path, noise, test_path=DEFAULT_IMG_PATH):
    
    def pre_process_fn(_x): return normilize(_x, 255) 
    def input_process_fn(_x): return gaussian(_x, is_training=True, mean=0, stddev=normilize(noise, 255))

    test_loader = DatasetFromFolder(test_path,
        pre_transform=pre_process_fn, use_cuda=USE_CUDA,
        inputs_transform=input_process_fn)
    test_loader = DataLoader(test_loader)

    model = LISTAConvDictADMM(
        num_input_channels=args['num_input_channels'],
        num_output_channels=args['num_output_channels'],
        kc=args['kc'], 
        ks=args['ks'],
        ista_iters=args['ista_iters'],
        iter_weight_share=args['iter_weight_share'],
    )
    common.load_eval(saved_model_path, model)
    #model.load_state_dict(torch.load(saved_model_path))
    if USE_CUDA:
        model = model.cuda()

    psnrs = []
    res_array = []
    idx = 0
    for img, img_n in test_loader:
        output = model(img_n)

        b = args['ks'] // 2

        np_img = to_np(img)[0,0,b:-b,b:-b]
        np_output = np.clip(to_np(output)[0,0,b:-b,b:-b], 0, 1)
        np_img_n = to_np(img_n)[0,0,b:-b,b:-b]

        psnrs.append(common.psnr(np_img, np_output, False))
        res_array.append((np_img, np_img_n, np_output))
        print('Test Image number {} psnr value {}'.format(idx, psnrs[-1]))
        idx += 1

    print('Avg psnr value is {}'.format(np.mean(psnrs)))
    return psnrs, res_array

def _test(args):
    mdl_p = args.saved_model_path
    tst_ims = args.test_im_path
    _args = arguments.load_args(args.saved_model_args_path)
    noise = _args.get('misc', {'noise': 20})['noise']
    model_args = _args['model_args']

    if not tst_ims == '':
        psnr, res = test(model_args, mdl_p, noise, tst_ims)
    else:
       psnr, res = test(model_args, mdl_p, noise)
    for idx, ims in enumerate(res):
        plot_res(ims[0], ims[1], ims[2], idx)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model_path', '-m', default='/home/hillel/projects/lista_admm/saved_models/trainSess_1/model_0.024123')
    parser.add_argument('--saved_model_args_path', '-a', default='/home/hillel/projects/lista_admm/saved_models/trainSess_1/params.txt')
    parser.add_argument('--test_im_path', default='')
    args = parser.parse_args()

    _test(args)

  
