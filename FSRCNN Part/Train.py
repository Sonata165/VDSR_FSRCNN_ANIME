import torch
import shutil
import matplotlib.pyplot as plt

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_pil_image, to_tensor

from Criteria import msssim, psnr
from CutImg import *
from Models import *
from DCSCN import *
from Constants import *
from Utils import *


def main():
    train('m4_epoch2000_temp', False)


def train(model_name, ctn=False):
    # 建立Tensorboard，清理日志log
    log_tmp_dir = 'log/log_temp/'
    writer = SummaryWriter(log_tmp_dir)

    # 准备数据
    dataset_train = MyDataset('data/cut_512/', 40)
    loader_train = Data.DataLoader(
        dataset=dataset_train,
        batch_size=10,
        shuffle=False,
    )
    total_steps = len(loader_train)

    dataset_valid = MyDataset('data/cut_512_test/', 40)
    loader_valid = Data.DataLoader(
        dataset=dataset_valid,
        batch_size=10,
        shuffle=False,
    )

    # 定义模型
    # model = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
    #                 single_conv_size=3, single_conv_group=1,
    #                 scale=4, activation=nn.LeakyReLU(0.1),
    #                 SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1)).to(device)
    if ctn == False:
        model = FSRCNN(scale_factor=4, num_channels=3, d=56, s=12, m=4).to(device)
    else:
        model = torch.load('models/' + model_name + '.pth')

    # Loss函数，优化器，EarlyStopping
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.Adam(model.last_part.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(model.)
    early_stopping = EarlyStopping(PATIENCE, verbose=True, delta=DELTA)

    summary(model, (3, 10, 10))

    for epoch in range(EPOCHS):
        # Training
        running_loss = 0.0
        for step, batch in enumerate(loader_train):
            lr_img, hr_img = batch

            # 算输出
            out = model.forward(lr_img)
            out = out.float()

            # 可视化
            if epoch % 50 == 0 and step == 0:
                print_patch(lr_img[0], name=str(epoch) + 'lr')
                print_patch(hr_img[0], name=str(epoch) + 'hr')
                print_patch(out[0], name=str(epoch) + 'out')

            # 算Loss
            loss = loss_func(out, hr_img)
            running_loss += loss.item()

            # 误差反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch: {} | Step {} / {} | Training Loss {:.4f}".format(epoch+1, step+1, total_steps, loss.item()))

        avg_train_loss = running_loss / total_steps

        # Validating
        with torch.no_grad():
            running_loss = 0.0
            running_ssim = 0.0
            running_psnr = 0.0
            for step, (lr_img, hr_img) in enumerate(loader_valid):
                # 算输出和Loss
                out = model.forward(lr_img)
                loss = loss_func(out, hr_img)
                running_loss += loss.item()

                # 算SSIM和PSNR
                running_ssim += msssim(out, hr_img).item()
                running_psnr += psnr(out, hr_img).item()

            avg_valid_loss = running_loss / len(loader_valid)
            avg_valid_ssim = running_ssim / len(loader_valid)
            avg_valid_psnr = running_psnr / len(loader_valid)

        print('Epoch: {} | Training Loss {:.4f} | Validating Loss {:.4f}'.format(
            epoch+1, avg_train_loss, avg_valid_loss,
        ))

        # Tensorboard可视化
        writer.add_scalars('Training Loss Graph', {'train_loss': avg_train_loss,
                                                   'validation_loss': avg_valid_loss}, epoch)
        writer.add_scalars('Validating SSIM Graph', {'ssim': avg_valid_ssim}, epoch)
        writer.add_scalars('Validating PSNR Graph', {'psnr': avg_valid_psnr}, epoch)

        # 用EarlyStopping判断有无过拟合。如果有，结束训练
        early_stopping(avg_valid_loss, model)
        if early_stopping.early_stop == True:
            print("Early Stopping!")
            break

    # 整理模型和日志
    shutil.move('checkpoint.pth', 'models/' + model_name + '.pth')

def print_patch(arr, name):
    arr = arr.data.cpu().numpy()
    arr = arr.transpose(1, 2, 0)
    arr = arr.astype(np.uint)
    plt.imshow(arr)
    # plt.show()
    plt.savefig('figure/' + name + '.png')

if __name__ == '__main__':
    main()
