import numpy as np
import cv2
from matplotlib import pyplot as plt

from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from Criteria import psnr_


def main():
    # resize()
    # denoise_128()
    denoise_final_result()


def resize():
    '''
    使用插值法放大图片, 对比试验
    :return:
    '''
    img = cv2.imread('data/cut_512_test/099.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.show()
    dst = cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4))
    #
    # dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    plt.subplot(121), plt.imshow(img)
    plt.subplot(122), plt.imshow(dst)
    # plt.show()

    # cv2.imwrite('lr.png', img)
    cv2.imwrite('dst99.png', dst)


def denoise():
    '''
    对指定的图片进行去噪处理
    '''
    img = cv2.imread('data/cut_512_results/099_4x.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    src = cv2.imread('data/cut_512_test/099.png')
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

    plt.subplot(121), plt.imshow(src)
    plt.subplot(122), plt.imshow(dst)
    # plt.subplot(122), plt.imshow(dst2)
    # plt.subplot(133), plt.imshow(dst2)
    plt.show()


def denoise_final_result():
    '''
    对results/下099_4x.png图片进行去噪处理
    :return:
    '''
    img = cv2.imread('results/099_4x.png')
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    cv2.imwrite('results/099_4x_denoise.png', dst)


def denoise_128():
    '''
    将网络对测试集128x128输出的图片进行进一步去噪处理, 得到最终结果
    '''
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
