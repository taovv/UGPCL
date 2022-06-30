import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms import transforms
from codes.builder import build_model
from codes.utils.utils import parse_yaml, Namespace

color1 = (102, 253, 204, 1.)
color2 = (255, 255, 102, 1.)
color3 = (255, 255, 255, 1.)


def _color(img, rgba=(102, 253, 204, 1.)):
    h, w, c = img.shape
    rgba = list(rgba)
    rgba[0] /= 255.
    rgba[1] /= 255.
    rgba[2] /= 255.
    img = img.tolist()
    for i in range(h):
        for j in range(w):
            if img[i][j] == [1., 1., 1., 1.]:
                img[i][j] = rgba
    return np.array(img)


def mixed_color(img1, img2):
    h, w, c = img1.shape
    for i in range(h):
        for j in range(w):
            if (img1[i][j] == [0., 0., 0., 1.]).all() or (img2[i][j] == [0., 0., 0., 1.]).all():
                img1[i][j] = img1[i][j] + img2[i][j]
                img1[i][j][-1] = 1.
            else:
                img1[i][j] = img1[i][j] * 0.5 + img2[i][j] * 0.5
    return np.array(img1)


def cal_dice(pred, gt):
    intersection = (pred * gt).sum()
    return (2 * intersection) / (pred.sum() + gt.sum()).item()


def show_pred_gt(model,
                 device='cuda',
                 img_size=(3, 256, 256),
                 datasets=r'D:\datasets\Breast\large\val',
                 output_path=r'../analyze_results/hardmseg'):
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model = model.to(device)
    img_path = os.path.join(datasets, 'images')
    mask_path = os.path.join(datasets, 'masks')
    names = os.listdir(img_path)
    dice_list = []
    for name in tqdm(names):
        if img_size[0] < 3:
            img = cv2.imread(os.path.join(img_path, name), cv2.IMREAD_GRAYSCALE)
            img = img[:np.newaxis]
        else:
            img = cv2.imread(os.path.join(img_path, name), cv2.IMREAD_COLOR)
        h, w, _ = img.shape
        img = cv2.resize(img, (img_size[1], img_size[2]))
        img = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).to(device, torch.float32)
        img = transforms.Normalize(std=std, mean=mean)(img)
        img = torch.cat([img, img], dim=0)
        with torch.no_grad():
            pred = model.inference(img)[0].argmax(0)
        pred = pred.view(img_size[1], img_size[1]).cpu().detach().numpy().astype(np.float32)
        mask = cv2.imread(os.path.join(mask_path, name.replace('jpg', 'png')), cv2.IMREAD_GRAYSCALE)
        dice = cal_dice(pred, mask / 255)
        dice_list.append(dice)
        mask = cv2.resize(mask, (img_size[1], img_size[2]))
        mask = (mask != 0).astype(np.float32)
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGRA)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
        plt.imsave(os.path.join(output_path, f'{int(dice * 10000)}_{name}'),
                   _color(mask, color3) * 0.5 + _color(pred, color1) * 0.5)
        # plt.imsave(os.path.join(output_path, name), mixed_color(_color(mask, color1), _color(pred, color3)))
    f = open(f'{output_path}/result.txt', 'w')
    for dice, name in zip(dice_list, names):
        f.write(f'{name}:\t{dice * 100:.2f}\n')
    f.write(f'average:\t{np.mean(dice_list) * 100:.2f}\n')
    f.close()


if __name__ == '__main__':
    args_dict = parse_yaml(r'F:\projects\PyCharmProjects\SSLSeg\configs\unet_breast_mri.yaml')
    model = build_model(args_dict['model'])
    # model = torch.nn.DataParallel(model)
    model.load_state_dict(
        torch.load(r'F:\projects\PyCharmProjects\SSLSeg\results\breast_mri\unet_breast_mri_0924194128\iter_6000.pth')[
            'state_dict'])
    show_pred_gt(model,
                 device='cuda',
                 img_size=(3, 256, 256),
                 datasets=r'F:\datasets\breast_mri\256x256\val',
                 output_path=r'F:\DeskTop\preds_unet_')
