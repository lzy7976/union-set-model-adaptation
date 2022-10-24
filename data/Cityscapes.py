from torch.utils import data
from PIL import Image
import numpy as np
import os


class Cityscapes(data.Dataset):
    def __init__(self, data_dir, split, psl_dir=None):
        self.data_dir = data_dir
        self.split = split
        self.psl_dir = psl_dir
        self.labelId2trainId = {7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7, 21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 31:16, 32:17, 33:18}

        self._get_samples()

    def _get_samples(self):
        self.sample_dicts = []
        city_list = os.listdir(os.path.join(self.data_dir, 'Cityscapes', 'leftImg8bit', self.split))
        for city in city_list:
            img_list = os.listdir(os.path.join(self.data_dir, 'Cityscapes', 'leftImg8bit', self.split, city))
            for img in img_list:
                sample_id = '_'.join(img.split('_')[:3])
                sample_dict = {
                    'id': sample_id,
                    'image': os.path.join(self.data_dir, 'Cityscapes', 'leftImg8bit', self.split, city, img),
                    'gt': os.path.join(self.data_dir, 'Cityscapes', 'gtFine', self.split, city, sample_id + '_gtFine_labelIds.png'),
                    'psl': os.path.join(self.psl_dir, sample_id + '_psl.png') if self.psl_dir is not None else None,
                }
                self.sample_dicts.append(sample_dict)

    def process_image(self, image):
        image = image.resize((1024, 512), Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]
        image -= np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)
        image = image.transpose((2, 0, 1))

        return image.copy()

    def process_label(self, label, to_trainId):
        label = np.asarray(label, np.float32)
        if to_trainId:
            label_trainId = np.zeros_like(label) + 255
            for k, v in self.labelId2trainId.items():
                label_trainId[label == k] = v
        else:
            label_trainId = label

        return label_trainId.copy()

    def __getitem__(self, index):
        sample_dict = self.sample_dicts[index]
        image = Image.open(sample_dict['image'])
        image = self.process_image(image)
        if sample_dict['psl'] is None:
            label = Image.open(sample_dict['gt'])
            label = self.process_label(label, True)
        else:
            label = Image.open(sample_dict['psl'])
            label = self.process_label(label, False)

        return sample_dict['id'], image, label

    def __len__(self):
        return len(self.sample_dicts)