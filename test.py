import os
import argparse
import torch
import numpy as np
import cv2

from tqdm import tqdm
from colorama import Fore
from prettytable import PrettyTable
from codes.builder import *
from codes.utils.utils import Namespace, parse_yaml

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_args(config_file):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default=config_file,
                        help='train config file path: xxx.yaml')
    parser.add_argument('--device', type=str, default='cuda:1' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    args_dict = parse_yaml(args.config)
    for key, value in Namespace(args_dict).__dict__.items():
        vars(args)[key] = value
    return args


def save_result(img,
                seg,
                file_path,
                opacity=0.5,
                palette=[[0, 0, 0], [0, 255, 255], [255, 106, 106], [255, 250, 240]]):
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    img = img * (1 - opacity) + color_seg * opacity
    img = img.astype(np.uint8)

    dir_name = os.path.abspath(os.path.dirname(file_path))
    os.makedirs(dir_name, exist_ok=True)
    cv2.imwrite(file_path, img)


def test(config_file, weights, save_pred=True):
    args = get_args(config_file)

    _, test_loader = build_dataloader(args.dataset, None)

    model = build_model(args.model)
    state_dict = torch.load(weights)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)

    print(F'\nModel: {model.__class__.__name__}')
    print(F'Weights file: {weights}')

    test_loader = tqdm(test_loader, desc='Testing', unit='batch',
                       bar_format='%s{l_bar}{bar}{r_bar}%s' % (Fore.LIGHTCYAN_EX, Fore.RESET))

    metrics = []
    if args.train.kwargs.metrics is not None:
        for metric_cfg in args.train.kwargs.metrics:
            metrics.append(_build_from_cfg(metric_cfg))

    test_res = None
    model.to(args.device)
    model.eval()
    i = 0
    for batch_data in test_loader:
        data, label = batch_data['image'].to(args.device), batch_data['label'].to(args.device)
        preds = model.inference(data)
        batch_metric_res = {}
        for metric in metrics:
            batch_metric_res[metric.name] = metric(preds, label)

        if save_pred:
            for j in range(preds.size(0)):
                save_result(
                    data[j].permute(1, 2, 0).cpu().numpy() * 255.,
                    preds[j][0].cpu().numpy(),
                    f'./shows/{i}.png',
                    opacity=1.0)
                i += 1

        if test_res is None:
            test_res = batch_metric_res
        else:
            for metric_name in test_res.keys():
                for key in test_res[metric_name].keys():
                    test_res[metric_name][key] += batch_metric_res[metric_name][key]
    for metric_name in test_res.keys():
        for key in test_res[metric_name].keys():
            test_res[metric_name][key] = test_res[metric_name][key] / len(test_loader)

        test_res_list = [_.cpu() for _ in test_res[metric_name].values()]
        test_res[metric_name]['Mean'] = np.mean(test_res_list[1:])

    test_table = PrettyTable()
    test_table.field_names = ['Metirc'] + list(list(test_res.values())[0].keys())
    for metric_name in test_res.keys():
        if metric_name in ['Dice', 'Jaccard', 'Acc', 'IoU', 'Recall', 'Precision']:
            temp = [float(format(_ * 100, '.2f')) for _ in test_res[metric_name].values()]
        else:
            temp = [float(format(_, '.2f')) for _ in test_res[metric_name].values()]
        test_table.add_row([metric_name] + temp)
    print(test_table.get_string())


def show_features(config_file, weights):
    args = get_args(config_file)

    _, test_loader = build_dataloader(args.dataset, None)

    model = build_model(args.model)
    state_dict = torch.load(weights)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)

    print(F'\nModel: {model.__class__.__name__}')
    print(F'Weights file: {weights}')

    model.to(args.device)
    model.eval()
    for batch_data in test_loader:
        data, label = batch_data['image'].to(args.device), batch_data['label'].to(args.device)
        preds = model.inference_features(data)
        features_ = preds['feats'][-3].permute(0, 2, 3, 1).contiguous().view(16 * 28 * 28, -1).cpu().detach().numpy()
        labels_ = torch.nn.functional.interpolate(label, (28, 28), mode='nearest')
        labels_ = labels_.permute(0, 2, 3, 1).contiguous().view(16 * 28 * 28, -1).squeeze(1).cpu().detach().numpy()

        features = features_[labels_ != 0]
        labels = labels_[labels_ != 0]
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(features)
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], str(labels[i]), color=plt.cm.Set1(labels[i]),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.show()
        exit()


if __name__ == '__main__':
    # test(r'weights/ugpcl_acdc_136/ugpcl_unet_r50.yaml',
    #      r'F:\DeskTop\UGPCL\weights\ugpcl_acdc_136\ugpcl_88.11.pth')
    show_features(r'weights/ugpcl_acdc_136/ugpcl_unet_r50.yaml',
                  r'F:\DeskTop\UGPCL\weights\ugpcl_acdc_136\ugpcl_88.11.pth')
