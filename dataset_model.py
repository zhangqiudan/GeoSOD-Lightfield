import os
from PIL import Image
import cv2
import torch
from torch.utils import data
from torchvision.transforms import functional as F
import numbers
import numpy as np
import random



class PairwiseImg(data.Dataset):
    def __init__(self):
        self.sal_root = '/data/lightfield/dataset/img_data/'
        self.sal_source = '/data/lightfield/dataset/img_data/train_list.lst'

        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]

        self.sal_num = len(self.sal_list)

    def __getitem__(self, item):
        sal_image_ori = load_image_ori(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[0]))
        sal_image_mid = load_image_mid(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[0]))
        sal_image_sml = load_image_sml(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[0]))
        sal_image1 = load_views_ori(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[1]))
        sal_image2 = load_views_ori(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[2]))
        sal_image3 = load_views_ori(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[3]))
        sal_image4 = load_views_mid(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[4]))
        sal_image5 = load_views_mid(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[5]))
        sal_image6 = load_views_mid(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[6]))
        sal_image7 = load_views_sml(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[7]))
        sal_image8 = load_views_sml(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[8]))
        sal_image9 = load_views_sml(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[9]))


        sal_depth = load_depth(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[10]))
        sal_label = load_sal_label(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[11]))

        sal_image_ori = torch.Tensor(sal_image_ori)
        sal_image_mid = torch.Tensor(sal_image_mid)
        sal_image_sml = torch.Tensor(sal_image_sml)
        sal_image1 = torch.Tensor(sal_image1)
        sal_image2 = torch.Tensor(sal_image2)
        sal_image3 = torch.Tensor(sal_image3)
        sal_image4 = torch.Tensor(sal_image4)
        sal_image5 = torch.Tensor(sal_image5)
        sal_image6 = torch.Tensor(sal_image6)
        sal_image7 = torch.Tensor(sal_image7)
        sal_image8 = torch.Tensor(sal_image8)
        sal_image9 = torch.Tensor(sal_image9)

        sal_depth = torch.Tensor(sal_depth)
        sal_label = torch.Tensor(sal_label)


        sample = {'Image_ori': sal_image_ori,'Image_mid': sal_image_mid,'Image_sml': sal_image_sml, 'MV_1': sal_image1, 'MV_2': sal_image2,'MV_3': sal_image3,
                'MV_4': sal_image4,'MV_5': sal_image5,'MV_6': sal_image6,
                'MV_7': sal_image7, 'MV_8': sal_image8,'MV_9': sal_image9,'depth_gt': sal_depth, 'gt_label': sal_label}
        return sample

    def __len__(self):
        return self.sal_num


class PairwiseImgTest(data.Dataset):
    def __init__(self, sal_mode='e'):

        if sal_mode == 'a':
            self.image_root = './data/Lytro/img/'
            self.image_source = './data/test_Lytro.lst'
            self.test_fold = './data/final_results/Lytro_test/'

        with open(self.image_source, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image_ori, name = load_image_test(
            os.path.join(self.image_root, self.image_list[item % self.image_num].split()[0]))
        image_mid = load_image_mid(os.path.join(self.image_root, self.image_list[item % self.image_num].split()[0]))
        image_sml = load_image_sml(os.path.join(self.image_root, self.image_list[item % self.image_num].split()[0]))
        MV_1 = load_views_ori(os.path.join(self.image_root, self.image_list[item % self.image_num].split()[1]))
        MV_2 = load_views_ori(os.path.join(self.image_root, self.image_list[item % self.image_num].split()[2]))
        MV_3 = load_views_ori(os.path.join(self.image_root, self.image_list[item % self.image_num].split()[3]))
        MV_4 = load_views_mid(os.path.join(self.image_root, self.image_list[item % self.image_num].split()[4]))
        MV_5 = load_views_mid(os.path.join(self.image_root, self.image_list[item % self.image_num].split()[5]))
        MV_6 = load_views_mid(os.path.join(self.image_root, self.image_list[item % self.image_num].split()[6]))
        MV_7 = load_views_sml(os.path.join(self.image_root, self.image_list[item % self.image_num].split()[7]))
        MV_8 = load_views_sml(os.path.join(self.image_root, self.image_list[item % self.image_num].split()[8]))
        MV_9 = load_views_sml(os.path.join(self.image_root, self.image_list[item % self.image_num].split()[9]))

        image_ori = torch.Tensor(image_ori)
        image_mid = torch.Tensor(image_mid)
        image_sml = torch.Tensor(image_sml)
        MV_1 = torch.Tensor(MV_1)
        MV_2 = torch.Tensor(MV_2)
        MV_3 = torch.Tensor(MV_3)
        MV_4 = torch.Tensor(MV_4)
        MV_5 = torch.Tensor(MV_5)
        MV_6 = torch.Tensor(MV_6)
        MV_7 = torch.Tensor(MV_7)
        MV_8 = torch.Tensor(MV_8)
        MV_9 = torch.Tensor(MV_9)
        return {'image_ori': image_ori, 'image_mid': image_mid, 'image_sml': image_sml, 'MV_1': MV_1, 'MV_2': MV_2,
                'MV_3': MV_3, 'MV_4': MV_4, 'MV_5': MV_5, 'MV_6': MV_6, 'MV_7': MV_7, 'MV_8': MV_8, 'MV_9': MV_9,
                'name': name}

    def save_folder(self):
        return self.test_fold

    def __len__(self):
        return self.image_num


# get the dataloader
def get_loader(batch_size, mode='train', num_thread=1, test_mode=0, sal_mode='e'):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = PairwiseImg()
    else:
        dataset = PairwiseImgTest(sal_mode=sal_mode)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
    return data_loader, dataset


def load_image_ori(pah):
    if not os.path.exists(pah):
        print('File Not Exists',pah)
    im = cv2.imread(pah)
    in_ = cv2.resize(im, (800, 600), interpolation=cv2.INTER_LINEAR)
    in_ = np.array(in_, dtype=np.float32)

    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2, 0, 1))

    return in_

def load_image_test(pah):
    if not os.path.exists(pah):
        print('File Not Exists')
    img_name = pah[56:-4]
    name = img_name + '.png'
    print('name:', name)

    im = cv2.imread(pah)
    in_ = cv2.resize(im, (800, 600), interpolation=cv2.INTER_LINEAR)
    in_ = np.array(in_, dtype=np.float32)

    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2, 0, 1))

    return in_, name

def load_image_mid(pah):
    if not os.path.exists(pah):
        print('File Not Exists')
    im = cv2.imread(pah)
    in_ = cv2.resize(im, (400, 300), interpolation=cv2.INTER_LINEAR)
    in_ = np.array(in_, dtype=np.float32)

    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2, 0, 1))

    return in_


def load_image_sml(pah):
    if not os.path.exists(pah):
        print('File Not Exists')

    im = cv2.imread(pah)
    in_ = cv2.resize(im, (200, 150), interpolation=cv2.INTER_LINEAR)
    in_ = np.array(in_, dtype=np.float32)


    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2, 0, 1))

    return in_

def load_views_ori(pah):
    if not os.path.exists(pah):
        print('File Not Exists:', pah)
        pah = pah[0:62] + '_LFR' + pah[62:]
        print('File Not Exists:', pah)

    im = cv2.imread(pah)
    in_ = cv2.resize(im, (800, 600), interpolation=cv2.INTER_LINEAR)
    in_ = np.array(in_, dtype=np.float32)

    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2, 0, 1))
    return in_


def load_views_mid(pah):
    if not os.path.exists(pah):
        print('File Not Exists:', pah)
        pah = pah[0:62] + '_LFR' + pah[62:]

    im = cv2.imread(pah)
    in_ = cv2.resize(im, (400, 300), interpolation=cv2.INTER_LINEAR)
    in_ = np.array(in_, dtype=np.float32)
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2, 0, 1))
    return in_


def load_views_sml(pah):
    if not os.path.exists(pah):
        print('File Not Exists:', pah)
        pah = pah[0:62] + '_LFR' + pah[62:]

    im = cv2.imread(pah)
    in_ = cv2.resize(im, (200, 150), interpolation=cv2.INTER_LINEAR)
    in_ = np.array(in_, dtype=np.float32)

    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2, 0, 1))
    return in_

def load_depth(pah):
    if not os.path.exists(pah):
        print('File Not Exists', pah)
    im = cv2.imread(pah)
    in_ = cv2.resize(im, (256, 256), Image.BILINEAR)
    label = np.array(in_, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:, :, 0]

    label = label / 255.
    label = label[np.newaxis, ...]
    return label


def load_sal_label(pah):
    if not os.path.exists(pah):
        print('File Not Exists', pah)
    im = cv2.imread(pah)
    in_ = cv2.resize(im, (256, 256), Image.BILINEAR)
    label = np.array(in_, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:, :, 0]
    label = label / 255.
    label = label[np.newaxis, ...]

    return label


