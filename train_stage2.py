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

    models = {}
    optimizers = {}
    for s in args.source:
        models[s] = Deeplab(num_classes=max(label_mappings[s].values()) + 1, restore_from=os.path.join(args.restore_path, f'{s}_best.pth'))
        optimizers[s] = SGD(models[s].optim_parameters(args), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
        models[s].to(args.gpu_id)
        for param in models[s].parameters():
            param.requires_grad = False
    model_fin = Deeplab(num_classes=args.num_classes, init_weights='DeepLab_init.pth')
    optimizer_fin = SGD(model_fin.optim_parameters(args), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    model_fin.to(args.gpu_id)

    data_loader = get_dataloader(args.data_dir, args.target, args.batch_size, psl_dir=args.pseudo_label_path, split='train')
    data_iter = iter(data_loader)

    best_mIoU = 0
    start_time = time.time()
    for i in range(args.num_steps):
        model_fin.train()
        model_fin.adjust_learning_rate(args, optimizer_fin, i)
        optimizer_fin.zero_grad()
        for model, optimizer in zip(models.values(), optimizers.values()):
            model.train()
            model.adjust_learning_rate(args, optimizer, i)
            optimizer.zero_grad()

        try:
            _, image, label = next(data_iter)
        except:
            data_iter = iter(data_loader)
            _, image, label = next(data_iter)

        mapped_labels = {}
        for s in args.source:
            if args.label_setting == 'f':
                mapped_labels[s] = label.long().to(device)
            else:
                mapped_label = torch.zeros_like(label) + 255
                for k, v in label_mappings[s].items():
                    mapped_label[label == k] = v
                mapped_labels[s] = mapped_label.long().to(device)
        image = image.to(device)
        label = label.long().to(device)
        n, _, h, w = image.size()

        losses = {}

        # Training whole final model

        for model in models.values():
            for param in model.layer5.parameters():
                param.requires_grad = True

        feat = model_fin(image, feat_only=True)
        preds = {s: models[s](image, feat=feat) for s in args.source}
        pred_ens = torch.zeros((n, args.num_classes, h, w), device=device)
        cnt = torch.zeros((1, args.num_classes, 1, 1), device=device)
        for s in args.source:
            for k, v in label_mappings[s].items():
                if v > 0 or args.label_setting == 'f':
                    pred_ens[:, k] += preds[s][:, v]
                    cnt[:, k] += 1
        assert (cnt == 0).sum() == 0
        pred_ens /= cnt
        loss_psl = F.cross_entropy(pred_ens, label, ignore_index=255)
        loss_mse = -(F.softmax(pred_ens, 1) ** 2).sum() / (h * w)

        loss_who = loss_psl + loss_mse
        loss_who.backward()
        losses['loss_who'] = loss_who.item()

        optimizer_fin.step()
        optimizer_fin.zero_grad()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad()

        # Training only classifiers

        losses['loss_cls'] = []
        for s in args.source:
            pred = models[s](image)
            loss_cls = F.cross_entropy(pred, mapped_labels[s], ignore_index=255)
            loss_cls.backward()
            losses['loss_cls'].append(loss_cls.item())

        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad()

        # Training only final backbone

        for model in models.values():
            for param in model.layer5.parameters():
                param.requires_grad = False

        loss_bak = 0
        feat = model_fin(image, feat_only=True)
        for s in args.source:
            pred = models[s](image)
            pred_fin = models[s](image, feat=feat)
            loss_bak += (F.kl_div(torch.log(F.softmax(pred_fin, 1).clamp(min=1e-10)), F.softmax(pred, 1), reduction='none')).sum() / (h * w)
        loss_bak /= len(args.source)
        loss_bak.backward()
        losses['loss_bak'] = loss_bak.item()

        optimizer_fin.step()
        optimizer_fin.zero_grad()

        if (i + 1) % args.save_freq == 0:
            if not os.path.exists(args.checkpoint_path):
                os.mkdir(args.checkpoint_path)
            torch.save(model_fin.state_dict(), os.path.join(args.checkpoint_path, f'final_{i + 1}.pth'))
            for s in args.source:
                torch.save(models[s].state_dict(), os.path.join(args.checkpoint_path, f'{s}_{i + 1}.pth'))

        if (i + 1) % args.print_freq == 0:
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, args.num_steps)
            for k, v in losses.items():
                if isinstance(v, list):
                    log += ", {}:".format(k)
                    for l in v:
                        log += " {:.4f}".format(l)
                else:
                    log += ", {}: {:.4f}".format(k, v)
            print(log)

        if (i + 1) % args.test_freq == 0:
            model_fin.eval()
            for model in models.values():
                model.eval()

            test_loader = get_dataloader(args.data_dir, args.target, args.batch_size, split='val')
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
            print('Final model mean IoU: {:.1f}'.format(IoU.mean()))
            if best_mIoU < IoU.mean():
                best_mIoU = IoU.mean()
                best_iter = i + 1
                torch.save(model_fin.state_dict(), os.path.join(args.checkpoint_path, 'final_best.pth'))
                for s in args.source:
                    torch.save(models[s].state_dict(), os.path.join(args.checkpoint_path, f'{s}_best.pth'))
    print('Best mean IoU {:.1f} at iteration {}'.format(best_mIoU, best_iter))


if __name__ == '__main__':
    main()
