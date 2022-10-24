import torch
import torch.nn.functional as F

import os
import numpy as np
from PIL import Image

from options import load_psl_options
from data import get_dataloader
from deeplab import Deeplab
from utils import get_label_mappings


def main():
    args = load_psl_options()
    device = torch.device('cuda:{}'.format(args.gpu_id))
    if args.label_setting == 'f':
        label_mappings = {s: {i: i for i in range(args.num_classes)} for s in args.source}
    else:
        label_mappings = get_label_mappings(args.source, args.label_setting)

    models = {}
    for s in args.source:
        models[s] = Deeplab(num_classes=max(label_mappings[s].values()) + 1, restore_from=os.path.join(args.restore_path, f'{s}_best.pth'))
        models[s].to(args.gpu_id)
        models[s].eval()

    dataloader = get_dataloader(args.data_dir, args.target, 1, split='train')

    labels = np.zeros((len(dataloader), 512, 1024))
    probs = np.zeros((len(dataloader), 512, 1024))
    img_ids = []
    with torch.no_grad():
        for i, [img_id, image, _] in enumerate(dataloader):
            preds = []
            for s in args.source:
                if args.stage == 1:
                    pred = models[s](image.to(device))
                    pred = F.softmax(pred, 1)
                    if args.label_setting == 'f':
                        preds.append(pred)
                    else:
                        mapped_pred = torch.zeros((1, args.num_classes, 512, 1024), device=device)
                        for k, v in label_mappings[s].items():
                            if v > 0:
                                mapped_pred[0, k] = pred[0, v]
                            else:
                                mapped_pred[0, k] = pred[0, v] / list(label_mappings[s].values()).count(0)
                        preds.append(mapped_pred)
                elif args.stage == 2:
                    feat = models[s](image.to(device), feat_only=True)
                    pred_ens = torch.zeros((1, args.num_classes, 512, 1024), device=device)
                    cnt = torch.zeros((1, args.num_classes, 1, 1), device=device)
                    for s_ in args.source:
                        pred = models[s_](image, feat=feat)
                        for k, v in label_mappings[s_].items():
                            if v > 0 or args.label_setting == 'f':
                                pred_ens[:, k] += pred[:, v]
                                cnt[:, k] += 1
                    pred_ens /= cnt
                    pred_ens = F.softmax(pred_ens, 1)
                    preds.append(pred_ens)
            mean_pred = np.array(torch.mean(torch.cat(preds, 0), 0).cpu())
            labels[i] = np.argmax(mean_pred, 0)
            probs[i] = np.max(mean_pred, 0)
            img_ids.append(img_id[0])

    thresholds = np.zeros(args.num_classes)
    for i in range(args.num_classes):
        probs_ = probs[labels == i]
        if len(probs_) > 0:
            probs_ = np.sort(probs_)
            thresholds[i] = probs_[round(len(probs_) * 0.5)]
    thresholds[thresholds > 0.9] = 0.9

    if not os.path.exists(args.pseudo_label_path):
        os.mkdir(args.pseudo_label_path)

    for i in range(len(dataloader)):
        label = labels[i].copy()
        for j in range(args.num_classes):
            label[(probs[i] < thresholds[j]) * (labels[i] == j)] = 255
        label = np.asarray(label, dtype=np.uint8)
        label = Image.fromarray(label)
        label.save(os.path.join(args.pseudo_label_path, f'{img_ids[i]}_psl.png'))
    
    
if __name__ == '__main__':
    main()