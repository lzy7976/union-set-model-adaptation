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

    data_loader = get_dataloader(args.data_dir, args.target, args.batch_size, psl_dir=args.pseudo_label_path, split='train')
    data_iter = iter(data_loader)

    best_mIoU = 0
    start_time = time.time()
    for i in range(args.num_steps):
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

        # Training with pseudo labels

        losses['loss_psl'] = []
        for s in args.source:
            pred, feat = models[s](image, extract_feat=True)
            loss_psl_sin = F.cross_entropy(pred, mapped_labels[s], ignore_index=255)
            loss_mse_sin = -(F.softmax(pred, 1) ** 2).sum() / (h * w)

            preds = {s_: models[s_](image, feat=feat) for s_ in args.source}
            pred_ens = torch.zeros((n, args.num_classes, h, w), device=device)
            cnt = torch.zeros((1, args.num_classes, 1, 1), device=device)
            for s_ in args.source:
                for k, v in label_mappings[s_].items():
                    if v > 0 or args.label_setting == 'f':
                        pred_ens[:, k] += preds[s_][:, v]
                        cnt[:, k] += 1
            assert (cnt == 0).sum() == 0
            pred_ens /= cnt
            loss_psl_ens = F.cross_entropy(pred_ens, label, ignore_index=255)
            loss_mse_ens = -(F.softmax(pred_ens, 1) ** 2).sum() / (h * w)

            loss_psl = loss_psl_sin + loss_psl_ens + loss_mse_sin + loss_mse_ens
            loss_psl.backward()
            losses['loss_psl'].append(loss_psl.item())

        # Training with cross-model consistency

        indices = [j for j in range(len(args.source))]
        while indices == [j for j in range(len(args.source))]:
            random.shuffle(indices)
        rand_match = {args.source[j]: args.source[indices[j]] for j in range(len(args.source))}

        feats = {s: models[s](image, feat_only=True) for s in args.source}
        pred_ens = torch.zeros((n, args.num_classes, h, w), device=device)
        pred_ens_cm = torch.zeros((n, args.num_classes, h, w), device=device)
        cnt = torch.zeros((1, args.num_classes, 1, 1), device=device)
        preds_cm = {}
        for s in args.source:
            pred = models[s](image, feat=feats[s])
            pred_cm = models[s](image, feat=feats[rand_match[s]])
            for k, v in label_mappings[s].items():
                if v > 0 or args.label_setting == 'f':
                    pred_ens[:, k] += pred[:, v]
                    pred_ens_cm[:, k] += pred_cm[:, v]
                    cnt[:, k] += 1
            preds_cm[s] = pred_cm
        pred_ens /= cnt
        pred_ens_cm /= cnt
        loss_cmc_1 = (F.softmax(pred_ens, 1) - F.softmax(pred_ens_cm, 1)).abs().mean()

        loss_cmc_2 = 0
        cnt = 0
        for c in range(args.num_classes):
            preds_c = []
            for s in args.source:
                if label_mappings[s][c] > 0 or args.label_setting == 'f':
                    preds_c.append(preds_cm[s][:, label_mappings[s][c]].unsqueeze(0))
            if len(preds_c) > 1:
                cnt += 1
                mean_pred_c = torch.mean(torch.cat(preds_c, 0), 0)
                for s in args.source:
                    if label_mappings[s][c] > 0 or args.label_setting == 'f':
                        loss_cmc_2 += (preds_cm[s][:, label_mappings[s][c]] - mean_pred_c).abs().mean()
        if cnt > 0:
            loss_cmc_2 /= cnt

        loss_cmc = loss_cmc_1 + loss_cmc_2
        loss_cmc.backward()
        losses['loss_cmc'] = loss_cmc.item()

        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad()

        # Training only classifiers

        losses['loss_cls'] = []
        feats = {s: models[s](image, feat_only=True).detach() for s in args.source}
        for s in args.source:
            preds = {s_: models[s](image, feat=feats[s_]) for s_ in args.source}
            loss_cls = F.cross_entropy(preds[s], mapped_labels[s], ignore_index=255)
            for s_ in args.source:
                if s_ != s:
                    loss_cls += -(F.softmax(preds[s_], 1) - F.softmax(preds[s], 1)).abs().mean() / (len(args.source) - 1)
            loss_cls.backward()
            losses['loss_cls'].append(loss_cls.item())

        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad()

        # Training only backbones

        for model in models.values():
            for param in model.layer5.parameters():
                param.requires_grad = False

        losses['loss_bak'] = []
        for s in args.source:
            feat = models[s](image, feat_only=True)
            preds = {s_: models[s_](image, feat=feat) for s_ in args.source}

            loss_bak_1 = 0
            for s_ in args.source:
                if s_ != s:
                    loss_bak_1 += F.cross_entropy(preds[s_], mapped_labels[s_], ignore_index=255)
                    loss_bak_1 += -(F.softmax(preds[s_], 1) ** 2).sum() / (h * w)
            loss_bak_1 /= (len(args.source) - 1)

            loss_bak_2 = 0
            cnt = 0
            for c in range(args.num_classes):
                preds_c = []
                for s_ in args.source:
                    if label_mappings[s_][c] > 0 or args.label_setting == 'f':
                        preds_c.append(preds[s_][:, label_mappings[s_][c]].unsqueeze(0))
                if len(preds_c) > 1:
                    cnt += 1
                    mean_pred_c = torch.mean(torch.cat(preds_c, 0), 0)
                    for s_ in args.source:
                        if label_mappings[s_][c] > 0 or args.label_setting == 'f':
                            loss_bak_2 += (preds[s_][:, label_mappings[s_][c]] - mean_pred_c).abs().mean()
            if cnt > 0:
                loss_bak_2 /= cnt

            loss_bak = loss_bak_1 + loss_bak_2
            loss_bak.backward()
            losses['loss_bak'].append(loss_bak.item())

        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad()

        for model in models.values():
            for param in model.layer5.parameters():
                param.requires_grad = True

        if (i + 1) % args.save_freq == 0:
            if not os.path.exists(args.checkpoint_path):
                os.mkdir(args.checkpoint_path)
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
            for model in models.values():
                model.eval()
            mIoUs = []
            for s in args.source:
                test_loader = get_dataloader(args.data_dir, args.target, args.batch_size, split='val')
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
                print('Source domain: {}, mean IoU: {:.1f}'.format(s, IoU.mean()))
                mIoUs.append(IoU.mean())
            if best_mIoU < sum(mIoUs) / len(mIoUs):
                best_mIoU = sum(mIoUs) / len(mIoUs)
                best_iter = i + 1
                for s in args.source:
                    torch.save(models[s].state_dict(), os.path.join(args.checkpoint_path, f'{s}_best.pth'))
    print('Best mean IoU {:.1f} at iteration {}'.format(best_mIoU, best_iter))


if __name__ == '__main__':
    main()
