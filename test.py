import torch
import torch.nn.functional as F
from torch.optim import SGD

import os
import time
import datetime
import random
import numpy as np

from options import load_test_options
from data import get_dataloader
from deeplab import Deeplab
from utils import get_label_mappings


def main():
    args = load_test_options()
    device = torch.device('cuda:{}'.format(args.gpu_id))
    if args.label_setting == 'f':
        label_mappings = {s: {i: i for i in range(args.num_classes)} for s in args.source}
    else:
        label_mappings = get_label_mappings(args.source, args.label_setting)

    models = {}
    optimizers = {}
    for s in args.source:
        models[s] = Deeplab(num_classes=max(label_mappings[s].values()) + 1, restore_from=os.path.join(args.restore_path, f'{s}_best.pth'))
        models[s].to(args.gpu_id)
        models[s].eval()
    if args.stage == 2:
        model_fin = Deeplab(num_classes=args.num_classes, restore_from=os.path.join(args.restore_path, 'final_best.pth'))
        model_fin.to(args.gpu_id)
        model_fin.eval()

    test_loader = get_dataloader(args.data_dir, args.target, 1, split='val')

    if args.stage == 1:
        IoUs = []
        for s in args.source:
            hist = np.zeros((args.num_classes, args.num_classes))
            with torch.no_grad():
                for _, image, label in test_loader:
                    image = image.to(device)
                    label = np.array(label).flatten().astype(int)
                    feat = models[s](image, feat_only=True)
                    pred_ens = torch.zeros((image.size()[0], args.num_classes, image.size()[2], image.size()[3]), device=device)
                    cnt = torch.zeros((1, args.num_classes, 1, 1), device=device)
                    for s_ in args.source:
                        pred = models[s_](image, feat=feat)
                        for k, v in label_mappings[s_].items():
                            if v > 0 or args.label_setting == 'f':
                                pred_ens[:, k] += pred[:, v]
                                cnt[:, k] += 1
                    pred_ens /= cnt
                    pred_ens = F.interpolate(pred_ens, (1024, 2048), mode='bilinear', align_corners=True)
                    pred_ens = np.array(pred_ens.cpu())
                    pred_ens = np.argmax(pred_ens, 1).flatten()
                    hist += np.bincount(args.num_classes * label[label != 255] + pred_ens[label != 255], minlength=args.num_classes ** 2).reshape(args.num_classes, args.num_classes)
            IoU = np.diag(hist) / (np.sum(hist, 0) + np.sum(hist, 1) - np.diag(hist)) * 100
            IoUs.append(np.expand_dims(IoU, 0))
        IoU = np.mean(np.concatenate(IoUs, 0), 0)
    elif args.stage == 2:
        hist = np.zeros((args.num_classes, args.num_classes))
        with torch.no_grad():
            for _, image, label in test_loader:
                image = image.to(device)
                label = np.array(label).flatten().astype(int)
                feat = model_fin(image, feat_only=True)
                pred_ens = torch.zeros((image.size()[0], args.num_classes, image.size()[2], image.size()[3]), device=device)
                cnt = torch.zeros((1, args.num_classes, 1, 1), device=device)
                for s in args.source:
                    pred = models[s](image, feat=feat)
                    for k, v in label_mappings[s].items():
                        if v > 0 or args.label_setting == 'f':
                            pred_ens[:, k] += pred[:, v]
                            cnt[:, k] += 1
                pred_ens /= cnt
                pred_ens = F.interpolate(pred_ens, (1024, 2048), mode='bilinear', align_corners=True)
                pred_ens = np.array(pred_ens.cpu())
                pred_ens = np.argmax(pred_ens, 1).flatten()
                hist += np.bincount(args.num_classes * label[label != 255] + pred_ens[label != 255], minlength=args.num_classes ** 2).reshape(args.num_classes, args.num_classes)
        IoU = np.diag(hist) / (np.sum(hist, 0) + np.sum(hist, 1) - np.diag(hist)) * 100

    print('Mean IoU: {:.1f}'.format(IoU.mean()))
    print('IoUs:', np.round(IoU, 1))


if __name__ == '__main__':
    main()
