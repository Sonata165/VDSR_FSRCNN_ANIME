import os
import numpy as np
import PIL.Image as Img

from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import to_tensor

from Constants import *

def main():
    # raw_to_png()
    # cut_all_imgs(512)
    cut_all_imgs(128)
    # dataset = MyDataset(train_data_path='data/cut_128/', patch_size=40)
    #
    # loader = Data.DataLoader(
    #     dataset=dataset,
    #     batch_size=1,
    #     shuffle=True,
    #     num_workers=0,
    # )
    #
    # for step, img in enumerate(loader):
    #     print('Step {} / {}'.format(step, 100))
    #     print(len(loader))
    #     print(img[0].shape)
    #     print(img[1].shape)
    #     print()


def rename_imgs():
    '''
    将raw下的文件按顺序命名
    '''
    path = 'data/raw/'
    filenames = os.listdir(path)
    cnt = 1
    for filename in filenames:
        os.rename(os.path.join(path, filename), os.path.join(path, '{:0>3d}'.format(cnt)))
        cnt += 1


def raw_to_png():
    '''
    将raw下的文件转换成png格式，保存在png_imgs文件夹下
    '''
    import cv2

    path = "data/raw/"
    out_path = 'data/png_imgs/'
    print(path)

    for filename in os.listdir(path):
        # img = cv2.imread(path + filename)
        # newfilename = filename + ".png"
        # cv2.imwrite(out_path + newfilename, img)
        img = Img.open(path + filename)
        img = img.convert('RGB')
        newfilename = filename + '.png'
        print(img.mode)
        img.save(out_path + newfilename)

def cut_all_imgs(size):
    '''
    将所有原始图片缩小到指定范围，作为训练数据
    :param size: 缩小后的图片的长边长度，要求是整数
    '''
    in_folder = 'data/png_imgs/'
    out_folder = 'data/cut_' + str(size) + '/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    files = os.listdir(in_folder)
    for file in files:
        filename = file.split('.')[0]
        im = Img.open(in_folder + file)
        out = resize_img(im, size)
        out.save(out_folder + filename + '.png')


def resize_img(im_t, limit):
    '''
    将一张图片缩小到指定范围
    :param im_t: 待缩小的图片
    :param limit: 缩小后的图片的长度（像素）
    :return: 缩小后的图片，PIL格式
    '''
    im = im_t.copy()
    (x, y) = im.size  # read image size
    # limit = TRAIN_IMG_SIZE // 2  # 降低分辨率后图片较长边的宽度
    if x >= y:
        x_s = limit
        y_s = round(y * x_s / x)
    else:
        y_s = limit
        x_s = round(x * y_s / y)
    out = im.resize((x_s, y_s), Img.ANTIALIAS)  # resize image with 抗锯齿

    # print('original size: ', x, y)
    # print('adjust size: ', x_s, y_s)

    return out


def img2arr(img):
    '''
    将PIL的图片格式转换成tensor，并从(w, h, c)转换成(c, w, h)
    :param img: PIL图片
    :return: 转换后的tensor
    '''
    arr = np.asarray(img)
    arr = arr.transpose(2, 0, 1)
    ret = torch.tensor(arr, dtype=torch.float).to(device)
    return ret


class MyDataset(Data.Dataset):
    def __init__(self, train_data_path, patch_size):
        '''
        构造方法
        '''
        self.folder = train_data_path
        self.img_augmenter = ImageAugment()
        self.filenames = os.listdir(train_data_path)
        self.img_nums = len(self.filenames)
        self.cropper = RandomCrop(size=patch_size)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img = Img.open(self.folder + filename)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_crop = self.cropper(img)

        hr_img = img_crop
        lr_img = resize_img(hr_img, 10)

        return img2arr(lr_img), img2arr(hr_img)

    def __len__(self):
        return self.img_nums


class ImageAugment:
    def __init__(self):
        pass


if __name__ == '__main__':
    main()
