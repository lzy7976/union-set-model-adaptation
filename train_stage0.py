import torch
import torch.nn.functional as F
from torch.optim import SGD

import os
import time
import datetime
import random
import numpy as np

from options import load_train_options
from data import get_dataloader
from deeplab import Deeplab
from utils import get_label_mappings


def main():
    args = load_train_options()
    device = torch.device('cuda:{}'.format(args.gpu_id))
    if args.label_setting == 'f':
        label_mappings = {s: {i: i for i in range(args.num_classes)} for s in args.source}
    else:
        label_mappings = get_label_mappings(args.source, args.label_setting)

    src_loaders = {}
    models = {}
    optimizers = {}
    for s in args.source:
        src_loaders[s] = get_dataloader(args.data_dir, s, args.batch_size)
        models[s] = Deeplab(num_classes=max(label_mappings[s].values()) + 1, init_weights='DeepLab_init.pth')
        optimizers[s] = SGD(models[s].optim_parameters(args), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
        models[s].to(args.gpu_id)

    start_time = time.time()
    for s in args.source:
        print(f'Training model of {s}')
        data_iter = iter(src_loaders[s])
        best_mIoU = 0

        for i in range(args.num_steps):
            models[s].train()
            models[s].adjust_learning_rate(args, optimizers[s], i)
            optimizers[s].zero_grad()

            try:
                image, label = next(data_iter)
            except:
                data_iter = iter(src_loaders[s])
                image, label = next(data_iter)
            image = image.to(device)
            if args.label_setting == 'f':
                mapped_label = label.long().to(device)
            else:
                mapped_label = torch.zeros_like(label)
                for k, v in label_mappings[s].items():
                    mapped_label[label == k] = v
                mapped_label = mapped_label.long().to(device)

            pred = models[s](image)
            loss = F.cross_entropy(pred, mapped_label, ignore_index=255)

            loss.backward()
            optimizers[s].step()
            losses = {}
            losses['loss'] = loss.item()

            if (i + 1) % args.save_freq == 0:
                if not os.path.exists(args.checkpoint_path):
                    os.mkdir(args.checkpoint_path)
                torch.save(models[s].state_dict(), os.path.join(args.checkpoint_path, f'{s}_{i + 1}.pth'))

            if (i + 1) % args.print_freq == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = 'Elapsed [{}], Iteration [{}/{}]'.format(et, i + 1, args.num_steps)
                for k, v in losses.items():
                    log += ", {}: {:.4f}".format(k, v)
                print(log)

            if (i + 1) % args.test_freq == 0:
                models[s].eval()
                test_loader = get_dataloader(args.data_dir, args.target, args.batch_size, split='val')
                num_classes = max(label_mappings[s].values()) + 1
                hist = np.zeros((num_classes, num_classes))
                with torch.no_grad():
                    for _, image, label in test_loader:
                        if args.label_setting == 'f':
                            mapped_label = label
                        else:
                            mapped_label = torch.zeros_like(label)
                            for k, v in label_mappings[s].items():
                                mapped_label[label == k] = v
                        image = image.to(device)
                        mapped_label = np.array(mapped_label).flatten().astype(int)
                        pred = models[s](image)
                        predict = F.interpolate(pred, (1024, 2048), mode='bilinear', align_corners=True)
                        predict = np.array(predict.cpu())
                        predict = np.argmax(predict, 1).flatten()
                        hist += np.bincount(num_classes * mapped_label[mapped_label != 255] + predict[mapped_label != 255], minlength=num_classes ** 2).reshape(num_classes, num_classes)
                IoU = np.diag(hist) / (np.sum(hist, 0) + np.sum(hist, 1) - np.diag(hist)) * 100
                if num_classes < args.num_classes:
                    IoU = IoU[1:]
                print('Source domain: {}, mean IoU: {:.1f}'.format(s, IoU.mean()))
                if best_mIoU < IoU.mean():
                    best_mIoU = IoU.mean()
                    best_iter = i + 1
                    torch.save(models[s].state_dict(), os.path.join(args.checkpoint_path, f'{s}_best.pth'))
        print('Source domain: {}, best mean IoU {:.1f} at iteration {}'.format(s, best_mIoU, best_iter))


if __name__ == '__main__':
    main()