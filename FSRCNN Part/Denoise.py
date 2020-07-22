import numpy as np
import cv2
from matplotlib import pyplot as plt

from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from Criteria import psnr_

def main():
    # resize()
    # compare()
    denoise_128()

def resize():
    img = cv2.imread('data/cut_512_test/099.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.show()
    dst = cv2.resize(img, (img.shape[1]*4, img.shape[0]*4))
    #
    # dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    plt.subplot(121), plt.imshow(img)
    plt.subplot(122), plt.imshow(dst)
    # plt.show()

    # cv2.imwrite('lr.png', img)
    cv2.imwrite('dst99.png', dst)

def compare():
    itp = Image.open('dst.png')
    itp_arr = to_tensor(itp)
    out = Image.open('data/cut_128_results/097_4x.png')
    out_arr = to_tensor(out)
    target = Image.open('data/cut_512_test/097.png')
    target_arr = to_tensor(target)

    psnr_itp = psnr_(itp_arr, target_arr)
    psnr_net = psnr_(out_arr, target_arr)
    print(psnr_itp, psnr_net)

def denoise():

    img = cv2.imread('data/cut_512_results/099_4x.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    src = cv2.imread('data/cut_512_test/099.png')
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

    # img2 = cv2.imread('data/cut_128_results/097_4x.png')
    # img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # dst2 = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 10)

    plt.subplot(121), plt.imshow(src)
    plt.subplot(122), plt.imshow(dst)
    # plt.subplot(122), plt.imshow(dst2)
    # plt.subplot(133), plt.imshow(dst2)
    plt.show()

def denoise_128():
    folder_in = 'data/cut_128_results/'
    folder_out = 'data/cut_128_denoise/'
    import os
    filenames = os.listdir(folder_in)
    for filename in filenames:
        img = cv2.imread(folder_in + filename)
        dst = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
        cv2.imwrite(folder_out + filename, dst)


if __name__ == '__main__':
    main()