from torch.utils.data import Dataset, DataLoader
import numpy as np
import imageio
import random
from os import listdir
from os.path import join
import torch


class DataGenerator1(Dataset):
    def __init__(self, cover_dir, stego_dir,blurred_dir,bs_dir):
        self.cover_path = cover_dir
        self.stego_path = stego_dir
        self.blurred_path=blurred_dir
        self.bs_path=bs_dir

        cover_list = listdir(cover_dir)
        stego_list = listdir(stego_dir)
        blurred_list=listdir(blurred_dir)
        bs_list=listdir(bs_dir)

        self.filename_list = cover_list

        cover_len = len(cover_list)
        stego_len = len(stego_list)
        blurred_len=len(blurred_list)
        bs_len=len(bs_list)

        assert cover_len != 0, "the cover directory:{} is empty!".format(cover_dir)
        assert stego_len != 0, "the stego directory:{} is empty!".format(stego_dir)
        assert blurred_len != 0, "the blurred directory:{} is empty!".format(blurred_dir)
        assert bs_len != 0, "the blurred&stego directory:{} is empty!".format(bs_dir)
        assert cover_len == stego_len, "the cover directory and stego directory don't have the same number files, " \
                                       "respectively: %d, %d" % (cover_len, stego_len)
        assert cover_len == blurred_len, "the cover directory and blurred directory don't have the same number files, " \
                                       "respectively: %d, %d" % (cover_len, blurred_len)
        assert cover_len == bs_len, "the cover directory and blurred&stego directory don't have the same number files, " \
                                       "respectively: %d, %d" % (cover_len, bs_len)

        img = imageio.imread(join(self.cover_path, self.filename_list[0]))
        self.img_shape = img.shape

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):
        batch = np.empty(shape=(4, self.img_shape[0], self.img_shape[1], 1), dtype='uint8')

        batch[0, :, :, 0] = imageio.imread(join(self.cover_path, self.filename_list[index]))
        batch[1, :, :, 0] = imageio.imread(join(self.stego_path, self.filename_list[index]))
        batch[2, :, :, 0] = imageio.imread(join(self.blurred_path,self.filename_list[index]))
        batch[3, :, :, 0] = imageio.imread(join(self.bs_path,self.filename_list[index]))

        label = torch.tensor([[0,0],[1,0],[0,1],[1,1]], dtype=torch.int64)

        rot = random.randint(0, 3)
        if random.random() < 0.5:
            return [torch.from_numpy(np.rot90(batch, rot, axes=[1, 2]).copy()), label]
        else:
            return [torch.from_numpy(np.flip(np.rot90(batch, rot, axes=[1, 2]).copy(), axis=2).copy()), label]


def generate_train_data(data_path, batch_size):
    train_data = DataGenerator1(data_path['train_cover'], data_path['train_stego'],data_path['train_blurred'],data_path['train_bs'])
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size['train'], shuffle=True, num_workers=2, drop_last=True)

    return train_loader


class DataGenerator2(Dataset):
    def __init__(self, cover_dir, stego_dir):
        self.cover_path = cover_dir
        self.stego_path = stego_dir

        cover_list = listdir(cover_dir)
        stego_list = listdir(stego_dir)
        self.filename_list = cover_list

        cover_len = len(cover_list)
        stego_len = len(stego_list)
        assert cover_len != 0, "the cover directory:{} is empty!".format(cover_dir)
        assert stego_len != 0, "the stego directory:{} is empty!".format(stego_dir)
        assert cover_len == stego_len, "the cover directory and stego directory don't have the same number files, " \
                                       "respectively： %d, %d" % (cover_len, stego_len)

        img = imageio.imread(join(self.cover_path, self.filename_list[0]))
        self.img_shape = img.shape

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):
        batch = np.empty(shape=(2, self.img_shape[0], self.img_shape[1], 1), dtype='uint8')

        batch[0, :, :, 0] = imageio.imread(join(self.cover_path, self.filename_list[index]))
        batch[1, :, :, 0] = imageio.imread(join(self.stego_path, self.filename_list[index]))

        label = torch.tensor([0, 1], dtype=torch.int64)

        rot = random.randint(0, 3)
        if random.random() < 0.5:
            return [torch.from_numpy(np.rot90(batch, rot, axes=[1, 2]).copy()), label]
        else:
            return [torch.from_numpy(np.flip(np.rot90(batch, rot, axes=[1, 2]).copy(), axis=2).copy()), label]


def generate_valid_data(data_path, batch_size):
    valid_data = DataGenerator2(data_path['valid_cover'], data_path['valid_stego'])
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size['valid'], drop_last=True)

    return valid_loader


def generate_test_data(data_path, batch_size):
    mytest_data = DataGenerator2(data_path['test_cover'], data_path['test_stego'])
    valid_loader = DataLoader(dataset=mytest_data, batch_size=batch_size, drop_last=True)

    return valid_loader