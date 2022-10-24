from torch.utils import data
from PIL import Image
import numpy as np
import os


class GTA5(data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labelId2trainId = {7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7, 21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 31:16, 32:17, 33:18}

        self._get_samples()

    def _get_samples(self):
        self.sample_dicts = []
        img_list = os.listdir(os.path.join(self.data_dir, 'GTA5', 'images'))
        for img in img_list:
            sample_dict = {
                'image': os.path.join(self.data_dir, 'GTA5', 'images', img),
                'label': os.path.join(self.data_dir, 'GTA5', 'labels', img),
            }
            self.sample_dicts.append(sample_dict)

    def process_image(self, image):
        image = image.resize((1280, 720), Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]
        image -= np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)
        image = image.transpose((2, 0, 1))

        return image.copy()

    def process_label(self, label):
        label = label.resize((1280, 720), Image.NEAREST)
        label = np.asarray(label, np.float32)
        label_trainId = np.zeros_like(label) + 255
        for k, v in self.labelId2trainId.items():
            label_trainId[label == k] = v

        return label_trainId.copy()

    def __getitem__(self, index):
        sample_dict = self.sample_dicts[index]
        image = Image.open(sample_dict['image'])
        image = self.process_image(image)
        label = Image.open(sample_dict['label'])
        label = self.process_label(label)

        return image, label

    def __len__(self):
        return len(self.sample_dicts)