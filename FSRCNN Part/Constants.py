import torch
import torch.utils.data as Data

TRAIN_IMG_SIZE = 512
TRAIN_DATA_PATH = 'data/cut_512/'

PATCH_SIZE = 20

EPOCHS = 2000

PATIENCE = 300
DELTA = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")