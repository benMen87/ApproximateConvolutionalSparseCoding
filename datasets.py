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

def load_img(filepath, convert='L'):
    img = np.array(Image.open(filepath).convert(convert).resize((320, 320)))
    img = Variable(torch.from_numpy(img[None,...]),requires_grad=False).float()
    return img

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, pre_transform, inputs_transform, use_cuda=True):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.pre_transform = pre_transform
        self.inputs_transform = inputs_transform
        self._use_cuda = use_cuda

    def __getitem__(self, index):
        _inputs = self.pre_transform(load_img(self.image_filenames[index]))
        if self.inputs_transform:
            _targets = self.inputs_transform(_inputs)
        if self._use_cuda:
            _inputs = _inputs.cuda()
            _targets = _targets.cuda()
        return _inputs, _targets

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

