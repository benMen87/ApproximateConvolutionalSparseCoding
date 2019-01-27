import torch
import numpy as np
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import arguments
import test_denoise

USE_CUDA = torch.cuda.is_available()

def my_subplot(data, dims, name, save_path):
    """Subplot dict/SC
    """
    plt.figure(figsize=(dims[0] - 1, dims[1] - 1))
    gs = gridspec.GridSpec(dims[0], dims[1], wspace=0.0, hspace=0.0,
        top=1.-0.5/(dims[0]+1), bottom=0.5/(dims[0]+1),
        left=0.5/(dims[1]+1), right=1-0.5/(dims[1]+1))

    for i in range(dims[0]):
        for j in range(dims[1]):
            f = data[0, i*dims[1] + j]
            ax = plt.subplot(gs[i,j])
            ax.imshow(f, cmap='gray', interpolation='bilinear')
            plt.xticks(range(0), [], color='white')
            plt.yticks(range(0), [], color='white')

    plt.savefig(os.path.join(save_path, name))
    plt.clf()

def plot_dict(model, save_path):
    """Plot covolutional dictionary
    """
    cd = model.conv_dictionary.cpu().numpy()

    in_ch, kc, k_rows, k_cols = cd.shape

    kers_per_col = int(np.sqrt(kc))
    kers_per_row = kers_per_col + (kc - kers_per_col**2)

    img_rows = kers_per_row * k_rows
    img_cols = kers_per_col * k_cols
    img_cd = np.reshape(cd, newshape=(img_rows, img_cols, in_ch)).squeeze()

    my_subplot(cd, [kers_per_row, kers_per_col], 'conv-dictionary', save_path)


def evaluate_csc(model, img_n, save_path, im_name):
    """Plot CSC
    """
    csc = model.forward_enc(img_n.unsqueeze(0)).detach().cpu().numpy()
    _, depth, rows, cols = csc.shape

    sc_per_col = int(np.sqrt(depth))
    sc_per_row = sc_per_col + (depth - sc_per_col**2)

    my_subplot(csc, [sc_per_row, sc_per_col], f'{im_name}-sparse-feature-maps', save_path)


def evaluate(args):

    test_args = args['test_args']
    model_args = args['model_args']
    model_path = test_args['load_path']
    tst_ims = test_args["testset_path"]
    noise = test_args['noise']

    model = test_denoise.restore_model(model_args, model_path)
    model = model.cuda() if USE_CUDA else model

    testset = test_denoise.create_famous_dataset(tst_ims, noise)
    log_dir = os.path.dirname(model_path)

    plot_dict(model, log_dir)
    evaluate_csc(model, testset[7][0], log_dir, testset.image_filenames[7])
    evaluate_csc(model, testset[0][0], log_dir, testset.image_filenames[0])

def main():
    """Run test on trained model.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--args_file', default='./my_args.json')
    args_file = parser.parse_args().arg_file

    _args = arguments.load_args(args_file)

    evaluate(_args)

if __name__ == '__main__':
    main()
