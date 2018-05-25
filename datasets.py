from __future__ import division
from torch.autograd import Variable
import torch.utils.data as data
import torch
import numpy as np
from os import listdir
from os.path import join
from PIL import Image



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)


class DatasetFromNPZ(data.Dataset):
    def __init__(self, npz_path, key, pre_transform, inputs_transform, use_cuda=True):
        super(DatasetFromNPZ, self).__init__()
        dataset = np.load(npz_path)
        self._use_cuda = use_cuda

        if key not in dataset:
            raise  ValueError('key is not valid for db {} valid keys are {}'.format(npz_path, dataset.keys()))
        self._targets = pre_transform(dataset[key])

        self._inputs_transform = inputs_transform

    def __getitem__(self, index):
        _targets = Variable(torch.from_numpy(self._targets[index]).float(), requires_grad=False)
        _inputs = self._inputs_transform(_targets)
        if self._use_cuda:
            _targets = _targets.cuda()
            _inputs = _inputs.cuda()

        return _inputs, _targets

    def __len__(self):
        return len(self._targets)

