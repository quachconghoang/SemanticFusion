import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils import data

import numpy as np
from cityscapes import Cityscapes
import yaml


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.35675976, 0.37380189, 0.3764753])
    std = np.array([0.32064945, 0.32098866, 0.32325324])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == "__main__":
    with open('kitti.yml') as fp:
        cfg = yaml.load(fp)
        cfg = cfg['data_kitti']

    d_mean = cfg['mean']
    d_std = cfg['std']
    d_rows = cfg['img_rows']
    d_cols = cfg['img_cols']

    # data_augmented = transforms.RandomCrop((d_rows, d_cols))

    data_transforms = {
        'rgb': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(d_mean, d_std)
        ]),
        'annotate': transforms.Compose([
            transforms.ToTensor()
        ]),
    }

    dataset = Cityscapes(cfg['path'], split='train', mode='gtFine', target_type='semantic',
                         transform= data_transforms['rgb'],
                         target_transform= data_transforms['annotate'],
                         augmentation_params= (369,738))
    # dataset = Cityscapes(cfg['path'], split='train', mode='gtFine', target_type='semantic')

    # img, anno = dataset[0]
    # img.show()
    # anno.show()
    # anno = anno.numpy().transpose(1,2,0).reshape(375,750)

    trainLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    img, anno = next(iter(trainLoader))