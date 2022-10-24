from data.Cityscapes import Cityscapes
from data.Synscapes import Synscapes
from data.GTA5 import GTA5
from data.SYNTHIA import SYNTHIA

from torch.utils import data


def get_dataloader(data_dir, dataset, batch_size, psl_dir=None, split=None, shuffle=True):
    assert dataset in ['Cityscapes', 'Synscapes', 'GTA5', 'SYNTHIA']

    if dataset == 'Cityscapes':
        dataset = Cityscapes(data_dir, split, psl_dir)
    elif dataset == 'Synscapes':
        dataset = Synscapes(data_dir)
    elif dataset == 'GTA5':
        dataset = GTA5(data_dir)
    elif dataset == 'SYNTHIA':
        dataset = SYNTHIA(data_dir)

    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return dataloader
