import torch
import common
from common import gaussian, normilize, nhwc_to_nchw, to_np
import numpy as np
from datasets import DatasetFromFolder
from torch.utils.data import DataLoader
import os
from convsparse_net import LISTAConvDict
import arguments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_IMG_PATH = '/data/hillel/data_sets/test_images/'
USE_CUDA = torch.cuda.is_available()


def plot_res(img, img_n, res, name, log_path):

    img = np.squeeze(img)
    img_n = np.squeeze(img_n)
    res = np.squeeze(res)

    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('clean')
    plt.subplot(132)
    plt.imshow(img_n, cmap='gray')
    plt.title('noise psnr {:.2f}'.format(common.psnr(img, img_n)))
    plt.subplot(133)
    plt.imshow(res, cmap='gray')
    plt.show()
    plt.title('clean psnr {:.2f}'.format(common.psnr(img, res)))
    plt.savefig(os.path.join(log_path, 'res_{}'.format(name)))
    plt.clf()


def test(args, saved_model_path, noise, testset_path):
    """Run predictable test
    """
    torch.manual_seed(7)

    def pre_process_fn(_x): return normilize(_x, 255)
    def input_process_fn(_x): return gaussian(_x, is_training=True, mean=0, stddev=normilize(noise, 255))

    test_loader = DatasetFromFolder(
        testset_path,
        pre_transform=pre_process_fn, use_cuda=USE_CUDA,
        inputs_transform=input_process_fn)
    test_loader = DataLoader(test_loader)

    model = LISTAConvDict(
        num_input_channels=args['num_input_channels'],
        num_output_channels=args['num_output_channels'],
        kc=args['kc'],
        ks=args['ks'],
        ista_iters=args['ista_iters'],
        iter_weight_share=args['iter_weight_share'],
        share_decoder=args['share_decoder']
    )
    common.load_eval(saved_model_path, model)

    if USE_CUDA:
        model = model.cuda()

    psnrs = []
    res_array = []
    idx = 0
    for img, img_n in test_loader:
        output, _ = model(img_n)

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

def _test(args_file):
    _args = arguments.load_args(args_file)
    test_args = _args['test_args']
    model_args = _args['model_args']

    model_path = test_args['load_path']
    tst_ims = test_args["testset_path"]
    noise = test_args['noise']

    log_dir = os.path.dirname(model_path)
    psnr, res = test(model_args, model_path, noise, tst_ims)
    for idx, ims in enumerate(res):
        plot_res(ims[0], ims[1], ims[2], idx, log_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg_file', default='./my_args.json')
    args_file = parser.parse_args().arg_file

    _test(args_file)

  
