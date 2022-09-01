import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam, SGD
from torch.autograd import Variable
from torch.backends import cudnn
from GeoModel import build_model, weights_init
import scipy.misc as sm
import numpy as np
import os
import torchvision.utils as vutils
import cv2
import torch.nn.functional as F
import math
import time
import sys
import PIL.Image
import scipy.io
import os
import logging

#EPSILON = 1e-8
p = OrderedDict()

base_model_cfg = 'resnet'
p['lr_bone'] = 5e-5  # Learning rate resnet:5e-5
p['lr_branch'] = 0.025  # Learning rate
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.90  # Momentum
lr_decay_epoch = [15, 24]  # [6, 9], now x3 #15
nAveGrad = 10  # Update the weights once in 'nAveGrad' forward passes
showEvery = 50
tmp_path = 'temp/tmp_see'


class Solver(object):
    def __init__(self, train_loader, test_loader, config, save_fold=None):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.save_fold = save_fold
        self.mean = torch.Tensor([123.68, 116.779, 103.939]).view(3, 1, 1) / 255.
        # inference: choose the side map (see paper)
        if config.visdom:
            self.visual = Viz_visdom("trueUnify", 1)
        self.build_model()
        if self.config.pre_trained: self.net.load_state_dict(torch.load(self.config.pre_trained))
        if config.mode == 'train':
            self.log_output = open("./run-model/logs/log_light_1.txt", 'w')
        else:
            print('Loading pre-trained model from %s...' % self.config.model)
            self.net_bone.load_state_dict(torch.load(self.config.model))
            self.net_bone.eval()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def get_params(self, base_lr):
        ml = []
        for name, module in self.net_bone.named_children():
            print(name)
            if name == 'loss_weight':
                ml.append({'params': module.parameters(), 'lr': p['lr_branch']})
            else:
                ml.append({'params': module.parameters()})
        return ml

    # build the network
    def build_model(self):
        self.net_bone = build_model(base_model_cfg)
        if self.config.cuda:
            self.net_bone = self.net_bone.cuda()

        self.net_bone.eval()
        self.net_bone.apply(weights_init)
        if self.config.mode == 'train':
            if self.config.load_bone == '':
                if base_model_cfg == 'resnet':
                    self.net_bone.base.load_state_dict(torch.load(self.config.resnet))
            if self.config.load_bone != '': self.net_bone.load_state_dict(torch.load(self.config.load_bone))

        self.lr_bone = p['lr_bone']
        self.lr_branch = p['lr_branch']
        self.optimizer_bone = Adam(filter(lambda p: p.requires_grad, self.net_bone.parameters()), lr=self.lr_bone,
                                   weight_decay=p['wd'])

        self.print_network(self.net_bone, 'trueUnify bone part')

    # update the learning rate
    def update_lr(self, rate):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * rate

    def test(self, test_mode=0):
        img_num = len(self.test_loader)
        index = 23 #models_9view_35mp_scale_aug
        datasetname = 'dutlfv2'
        name_t1 = 'GNN_model_' + str(index) + '_sf_' +datasetname + '/'

        if not os.path.exists(os.path.join(self.save_fold, name_t1)):
            os.mkdir(os.path.join(self.save_fold, name_t1))
        for i, data_batch in enumerate(self.test_loader):
            self.config.test_fold = self.save_fold
            print(self.config.test_fold)
            images_ori_, images_mid_, images_sml_, MV_1_, MV_2_, MV_3_,MV_4_, MV_5_, MV_6_, MV_7_, MV_8_, MV_9_, name = data_batch['image_ori'],data_batch['image_mid'],data_batch['image_sml'], data_batch['MV_1'], data_batch['MV_2'],data_batch['MV_3'], \
                                                  data_batch['MV_4'], data_batch['MV_5'], data_batch['MV_6'], data_batch['MV_7'], data_batch['MV_8'], data_batch['MV_9'], data_batch['name']

            with torch.no_grad():
                images_ori = Variable(images_ori_).cuda()
                images_mid = Variable(images_mid_).cuda()
                images_sml = Variable(images_sml_).cuda()
                MV_1 = Variable(MV_1_).cuda()
                MV_2 = Variable(MV_2_).cuda()
                MV_3 = Variable(MV_3_).cuda()
                MV_4 = Variable(MV_4_).cuda()
                MV_5 = Variable(MV_5_).cuda()
                MV_6 = Variable(MV_6_).cuda()
                MV_7 = Variable(MV_7_).cuda()
                MV_8 = Variable(MV_8_).cuda()
                MV_9 = Variable(MV_9_).cuda()

                print(images_ori.size())

                up_x1, up_x2, up_x3 = self.net_bone(images_ori, images_mid, images_sml, MV_1, MV_2, MV_3, MV_4, MV_5, MV_6, MV_7, MV_8, MV_9)


                pred = np.squeeze(torch.sigmoid(up_x1[0]).cpu().data.numpy())
                pred = 255 * pred
                cv2.imwrite(os.path.join(self.config.test_fold, name_t1, name[0]), pred)

        print('Test Done!')

    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        for epoch in range(self.config.epoch):
            r_sal_loss, r_sum_loss = 0, 0
            self.net_bone.zero_grad()
            for i, data_batch in enumerate(self.train_loader):

                sal_image_ori, sal_image_mid,sal_image_sml, sal_image1, sal_image2, sal_image3, sal_image4, sal_image5, sal_image6, sal_image7, sal_image8, sal_image9, depth_label, sal_label = \
                    data_batch['Image_ori'], data_batch['Image_mid'], data_batch['Image_sml'], data_batch['MV_1'], data_batch['MV_2'], data_batch['MV_3'], data_batch['MV_4'],data_batch['MV_5'], \
                    data_batch['MV_6'],data_batch['MV_7'],data_batch['MV_8'],data_batch['MV_9'],data_batch['depth_gt'],data_batch['gt_label']

                sal_image_ori, sal_image_mid,sal_image_sml, sal_image1, sal_image2, sal_image3, sal_image4, sal_image5, sal_image6,sal_image7, sal_image8, sal_image9,depth_label, sal_label = \
                    Variable(sal_image_ori),Variable(sal_image_mid),Variable(sal_image_sml), Variable(sal_image1), Variable(sal_image2), Variable(sal_image3),Variable(sal_image4), Variable(sal_image5), Variable(sal_image6),\
                    Variable(sal_image7), Variable(sal_image8), Variable(sal_image9),Variable(depth_label), Variable(sal_label)

                if self.config.cuda:
                    sal_image_ori, sal_image_mid, sal_image_sml, sal_image1, sal_image2, sal_image3 = sal_image_ori.cuda(),sal_image_mid.cuda(),sal_image_sml.cuda(), sal_image1.cuda(), sal_image2.cuda(), sal_image3.cuda()
                    sal_image4, sal_image5, sal_image6, sal_image7, sal_image8, sal_image9 = sal_image4.cuda(), sal_image5.cuda(), sal_image6.cuda(), sal_image7.cuda(), sal_image8.cuda(),sal_image9.cuda()
                    depth_label, sal_label = depth_label.cuda(), sal_label.cuda()

                up_x1, up_x2, up_x3 = self.net_bone(sal_image_ori,sal_image_mid, sal_image_sml, sal_image1, sal_image2, sal_image3,sal_image4, sal_image5, sal_image6,sal_image7, sal_image8, sal_image9)

                # sal part
                sal_loss1 = F.binary_cross_entropy_with_logits(up_x1, sal_label, reduction='sum') + structure_loss(up_x1, sal_label)
                sal_loss2 = F.binary_cross_entropy_with_logits(up_x2, sal_label, reduction='sum') + structure_loss(up_x2, sal_label)
                sal_loss3 = F.smooth_l1_loss(up_x3, depth_label, size_average=True)  + structure_loss(up_x3, depth_label)

                sal_loss = (sal_loss1 + sal_loss2 + sal_loss3) / (nAveGrad * self.config.batch_size)

                r_sal_loss += sal_loss.data
                loss = sal_loss
                r_sum_loss += loss.data
                loss.backward()
                aveGrad += 1

                if aveGrad % nAveGrad == 0:
                    self.optimizer_bone.step()
                    self.optimizer_bone.zero_grad()
                    aveGrad = 0

                if i % showEvery == 0:
                    print(
                            'epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %10.4f ' % (
                        epoch, self.config.epoch, i, iter_num,
                        r_sal_loss * (nAveGrad * self.config.batch_size) / showEvery))

                    print('Learning rate: ' + str(self.lr_bone))
                    r_sal_loss = 0

                if i % 200 == 0:
                    vutils.save_image(torch.sigmoid(up_x1[-1].data), tmp_path + '/iter%d-sal-0.jpg' % i,
                                      normalize=True, padding=0)

                    vutils.save_image(sal_image_ori.data, tmp_path + '/iter%d-sal-data.jpg' % i, padding=0)
                    vutils.save_image(sal_label.data, tmp_path + '/iter%d-sal-target.jpg' % i, padding=0)

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net_bone.state_dict(),
                           './run-model/models_save/epoch_%d_bone.pth' % (epoch + 1))

            if epoch in lr_decay_epoch:
                self.lr_bone = self.lr_bone * 0.1
                self.optimizer_bone = Adam(filter(lambda p: p.requires_grad, self.net_bone.parameters()),
                                           lr=self.lr_bone, weight_decay=p['wd'])

        torch.save(self.net_bone.state_dict(), './run-model/models_save/final_bone.pth')

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).sum()