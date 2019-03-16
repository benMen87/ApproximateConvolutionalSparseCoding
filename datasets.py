from __future__ import division
from torch.autograd import Variable
import torch.utils.data as data
from functools import partial
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath, convert='L'):
    img = np.array(Image.open(filepath).convert(convert))
    img = Variable(torch.from_numpy(img[None,...]),requires_grad=False).float()
    return img

def find_file_in_folder(folder, file_name):
    for f in os.listdir(folder):
        f_split = os.path.splitext(f)
        if file_name == f_split[0]:
            break
    else:
        return None
    return os.path.join(folder, f)


def get_images_from_file_names(img_dir, file_of_filenames):
    find_file = partial(find_file_in_folder, img_dir)
    file_names = []
    with open(file_of_filenames, 'r') as fp:
        for fname in fp:
            full_fname = find_file(fname.rstrip())

            if full_fname is None:
                raise ValueError(
                    f"missing file {fname} in folder {img_dir}"
                )
            file_names.append(full_fname)
    return file_names

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, pre_transform, inputs_transform,
                 file_of_filenames=None, use_cuda=True):
        """
        Dataset that each data is a file in folder
        There may be given a file that spesices what type of files to load.

        Args:
            image_dir(str): folder with data images for dataset
            pre_transform(fn): function on all data (such as basic scaling)
            inputs_transform(fn): function on input data such as shift adding
                noise etc.
            file_of_filenames(str, optional): path to file that contains names
            of specific files to load

            inputs_transform(fn):

        """
        super(DatasetFromFolder, self).__init__()
        self.pre_transform = pre_transform
        self.inputs_transform = inputs_transform
        self._use_cuda = use_cuda

        if file_of_filenames is None:
            self._image_filenames = [
                join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)
            ]
        else:
            self._image_filenames = get_images_from_file_names(image_dir,
                                                               file_of_filenames)


    @property
    def image_filenames(self):
        """Get list of file names
        """
        def fname(name):
            """Get file name from full path
            """
            return os.path.splitext(os.path.basename(name))[0]
        return list(map(fname, self._image_filenames))

    def __getitem__(self, index):
        _inputs = self.pre_transform(load_img(self._image_filenames[index]))
        if self.inputs_transform:
            _targets = self.inputs_transform(_inputs)
        if self._use_cuda:
            _inputs = _inputs.cuda()
            _targets = _targets.cuda()
        return _inputs, _targets

    def __len__(self):
        return len(self._image_filenames)

class DatasetFromNPZ(data.Dataset):
    def __init__(self, npz_path, key, pre_transform, inputs_transform, use_cuda=True):
        super(DatasetFromNPZ, self).__init__()
        dataset = np.load(npz_path)
        self._use_cuda = use_cuda

        if key not in dataset:
            raise ValueError('key is not valid for db {} valid keys are {}'
                             .format(npz_path, dataset.keys()))

        self._targets = pre_transform(dataset[key])
        print(len(self._targets))
        self._inputs_transform = inputs_transform

    def __getitem__(self, index):
        _targets = Variable(torch.from_numpy(self._targets[index]).float(), requires_grad=False)
        _inputs = self._inputs_transform(_targets)
        if self._use_cuda:
            _targets = _targets.cuda()
            _inputs = _inputs.cuda()

        return _targets, _inputs

    def __len__(self):
        return len(self._targets)

def debug():
    dset = DatasetFromFolder(
        '/data/VOCdevkit/VOC2010/JPEGImages',
        pre_transform=lambda x: x,
        inputs_transform=lambda x: x,
        file_of_filenames='./pascal2010_test_imgs.txt'
    )

if __name__ == '__main__':
    debug()
