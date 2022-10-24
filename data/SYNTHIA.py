from torch.utils import data
from PIL import Image
import imageio
imageio.plugins.freeimage.download()
import numpy as np
import os


class SYNTHIA(data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labelId2trainId = {3:0, 4:1, 2:2, 21:3, 5:4, 7:5, 15:6, 9:7, 6:8, 16:9, 1:10, 10:11, 17:12, 8:13, 18:14, 19:15, 20:16, 12:17, 11:18}

        self._get_samples()

    def _get_samples(self):
        self.sample_dicts = []
        img_list = os.listdir(os.path.join(self.data_dir, 'SYNTHIA', 'RGB'))
        for img in img_list:
            sample_dict = {
                'image': os.path.join(self.data_dir, 'SYNTHIA', 'RGB', img),
                'label': os.path.join(self.data_dir, 'SYNTHIA', 'GT', 'LABELS', img),
            }
            self.sample_dicts.append(sample_dict)

    def process_image(self, image):
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]
        image -= np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)
        image = image.transpose((2, 0, 1))

        return image.copy()

    def process_label(self, label):
        label = np.asarray(label, np.float32)[:,:,0]
        label_trainId = np.zeros_like(label) + 255
        for k, v in self.labelId2trainId.items():
            label_trainId[label == k] = v

        return label_trainId.copy()

    def __getitem__(self, index):
        sample_dict = self.sample_dicts[index]
        image = Image.open(sample_dict['image'])
        image = self.process_image(image)
        label = imageio.imread(sample_dict['label'], format='PNG-FI')
        label = self.process_label(label)

        return image, label

    def __len__(self):
        return len(self.sample_dicts)