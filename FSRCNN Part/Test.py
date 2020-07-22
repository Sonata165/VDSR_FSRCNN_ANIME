import os
import torch
# from prepare_images import *
from Models import *
from torchvision.utils import save_image
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from Constants import *
from Utils import *
from CutImg import img2arr
from Train import print_patch

from Criteria import psnr_, msssim


def main():
    evaluate()


def evaluate():
    folder_net = 'data/cut_128_results/'
    folder_den = 'data/cut_128_denoise/'
    folder_hr = 'data/cut_512_test/'
    filenames_net = os.listdir(folder_net)
    filenames_den = os.listdir(folder_den)
    filenames_hr = os.listdir(folder_hr)
    filenames = filenames_hr.copy()

    psnr_before = 0.0
    ssim_before = 0.0
    psnr_after = 0.0
    ssim_after = 0.0

    for i in range(len(filenames_hr)):
        img_net = Image.open(folder_net + filenames_net[i])
        img_den = Image.open(folder_den + filenames_den[i])
        img_hr = Image.open(folder_hr + filenames_hr[i])

        if img_hr.size[0] != img_net.size[0] or img_hr.size[1] != img_net.size[1]:
            img_hr = img_hr.resize((img_net.size[0], img_net.size[1]), Image.ANTIALIAS)
        img_hr.show()

        arr_net = to_tensor(img_net)
        arr_den = to_tensor(img_den)
        arr_hr = to_tensor(img_hr)

        psnr_before += psnr_(arr_net, arr_hr)
        psnr_after += psnr_(arr_den, arr_hr)

        arr_net = arr_net.unsqueeze(0)
        arr_den = arr_den.unsqueeze(0)
        arr_hr = arr_hr.unsqueeze(0)

        ssim_before += msssim(arr_net, arr_hr)
        ssim_after += msssim(arr_den, arr_hr)

    print(psnr_before/10, ssim_before/10, psnr_after/10, ssim_after/10)




def test_model():
    model = torch.load('models/m4_1000epoch.pth').to(device)
    # model = torch.load('models/FirstFSRCNN_m8.pth').to(device)
    # model = torch.load('models/FirstModel.pth').to(device)

    input_dir = 'data/cut_128_test/'
    output_dir = 'data/cut_128_results/'

    for files in os.listdir(input_dir):
        # load image
        print(files)
        img = Image.open(os.path.join(input_dir, files))
        if img.mode != 'RGB':
            img = img.convert("RGB")

        # 可视化输入图片
        img_t = img2arr(img).to(device)
        print_patch(img_t, name='shit' + 'lr')
        img_t = img_t.unsqueeze(0)

        # 算输出
        out = model(img_t).squeeze()

        # 可视化输出图片，保存
        norm_out = out.add(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(norm_out)
        im.save(os.path.join(output_dir, files[:-4] + '_' + '4x' + '.png'))


if __name__ == '__main__':
    main()
