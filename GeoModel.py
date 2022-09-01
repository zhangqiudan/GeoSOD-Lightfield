import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np
import units.ConvGRU2 as ConvGRU
from resnet import resnet50
import time


config_resnet = {'convert': [[64, 256, 512, 1024, 2048], [128, 256, 512, 512, 512]]}

class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        up0, up1, up2 = [], [], []
        for i in range(len(list_k[0])):
            up0.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1), nn.ReLU(inplace=True)))

        self.convert0 = nn.ModuleList(up0)

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl


class GeoModel(nn.Module):
    def __init__(self, all_channel=256, all_dim=60):
        super(GeoModel, self).__init__()

        self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        self.channel = all_channel
        self.dim = all_dim * all_dim
        self.gate = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.conv1 = nn.Sequential(nn.Conv2d(all_channel * 2, all_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.ConvGRU = ConvGRU.ConvGRUCell(all_channel, all_channel, all_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(all_channel)
        self.conv_fusion = nn.Conv2d(all_channel * 3, all_channel, kernel_size=3, padding=1, bias=True)
        self.relu_fusion = nn.ReLU(inplace=True)
        self.prelu = nn.ReLU(inplace=True)
        self.relu_m = nn.ReLU(inplace=True)

        self.sal_conv_f1 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(inplace=True))
        self.sal_conv_f2 = nn.Sequential(nn.Conv2d(768, 256, 3, 1, 1),nn.ReLU(inplace=True),
                                         nn.Conv2d(256, 1, 3, 1, 1))


        self.dis_conv1 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(inplace=True))
        self.dis_conv1_h2 = nn.Sequential(nn.Conv2d(768, 512, 3, 1, 1), nn.ReLU(inplace=True),
                                       nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(inplace=True))

        self.dis_conv2 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(inplace=True))
        self.dis_conv2_h3 = nn.Sequential(nn.Conv2d(768, 512, 3, 1, 1), nn.ReLU(inplace=True),
                                       nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(inplace=True))

        self.dis_conv3 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(inplace=True))
        self.dis_conv3_h4 = nn.Sequential(nn.Conv2d(768, 512, 3, 1, 1), nn.ReLU(inplace=True),
                                       nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(inplace=True))

        self.dis_conv4_f = nn.Sequential(nn.Conv2d(768, 512, 3, 1, 1), nn.ReLU(inplace=True),
                                       nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(inplace=True))

        self.dis_sod = nn.Conv2d(256, 1, 3, 1, 1)
        self.dis_dis = nn.Conv2d(256, 1, 3, 1, 1)
        self.propagate_layers = 5

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, list_x1_ori, list_x1_mid, list_x1_sml, list_x2, list_x3, list_x4, list_x5, list_x6, list_x7, list_x8, list_x9, list_x10,
                x_size):

        batch_num = list_x1_ori[4].size()[0]

        exemplars = self.conv1(list_x1_ori[4])
        querys = self.conv1(list_x2[4])
        query1s = self.conv1(list_x3[4])
        query2s = self.conv1(list_x4[4])

        exemplars_s = self.conv1(list_x1_mid[4])
        querys_s = self.conv1(list_x5[4])
        query1s_s = self.conv1(list_x6[4])
        query2s_s = self.conv1(list_x7[4])

        exemplars_t = self.conv1(list_x1_sml[4])
        querys_t = self.conv1(list_x8[4])
        query1s_t = self.conv1(list_x9[4])
        query2s_t = self.conv1(list_x10[4])

        x1s = torch.zeros(batch_num, 1, x_size[0], x_size[1]).cuda()
        x2s = torch.zeros(batch_num, 1, x_size[0], x_size[1]).cuda()
        x3s = torch.zeros(batch_num, 1, x_size[0], x_size[1]).cuda()

        for ii in range(batch_num):

            exemplar = exemplars[ii, :, :, :][None].contiguous().clone()
            query = querys[ii, :, :, :][None].contiguous().clone()
            query1 = query1s[ii, :, :, :][None].contiguous().clone()
            query2 = query2s[ii, :, :, :][None].contiguous().clone()

            exemplar_s = exemplars_s[ii, :, :, :][None].contiguous().clone()
            query_s = querys_s[ii, :, :, :][None].contiguous().clone()
            query1_s = query1s_s[ii, :, :, :][None].contiguous().clone()
            query2_s = query2s_s[ii, :, :, :][None].contiguous().clone()

            exemplar_t = exemplars_t[ii, :, :, :][None].contiguous().clone()
            query_t = querys_t[ii, :, :, :][None].contiguous().clone()
            query1_t = query1s_t[ii, :, :, :][None].contiguous().clone()
            query2_t = query2s_t[ii, :, :, :][None].contiguous().clone()

            for passing_round in range(self.propagate_layers):

                attention1 = self.conv_fusion(torch.cat([self.generate_attention(exemplar, query),
                                                         self.generate_attention(exemplar, query1),
                                                         self.generate_attention(exemplar, query2)],
                                                        1))
                attention2 = self.conv_fusion(torch.cat([self.generate_attention(query, exemplar),
                                                         self.generate_attention(query, query1),
                                                         self.generate_attention(query, query2)], 1))

                attention3 = self.conv_fusion(torch.cat([self.generate_attention(query1, exemplar),
                                                         self.generate_attention(query1, query),
                                                         self.generate_attention(query1, query2)], 1))

                attention4 = self.conv_fusion(torch.cat([self.generate_attention(query2, exemplar),
                                                         self.generate_attention(query2, query),
                                                         self.generate_attention(query2, query1)], 1))

                h_v1 = self.ConvGRU(attention1, exemplar)
                h_v2 = self.ConvGRU(attention2, query)
                h_v3 = self.ConvGRU(attention3, query1)
                h_v4 = self.ConvGRU(attention4, query2)

                exemplar = h_v1.clone()
                query = h_v2.clone()
                query1 = h_v3.clone()
                query2 = h_v4.clone()

                attention1_s = self.conv_fusion(torch.cat([self.generate_attention(exemplar_s, query_s),
                                                           self.generate_attention(exemplar_s, query1_s),
                                                           self.generate_attention(exemplar_s, query2_s)],
                                                          1))  # message passing with concat operation
                attention2_s = self.conv_fusion(torch.cat([self.generate_attention(query_s, exemplar_s),
                                                           self.generate_attention(query_s, query1_s),
                                                           self.generate_attention(query_s, query2_s)], 1))

                attention3_s = self.conv_fusion(torch.cat([self.generate_attention(query1_s, exemplar_s),
                                                           self.generate_attention(query1_s, query_s),
                                                           self.generate_attention(query1_s, query2_s)], 1))

                attention4_s = self.conv_fusion(torch.cat([self.generate_attention(query2_s, exemplar_s),
                                                           self.generate_attention(query2_s, query_s),
                                                           self.generate_attention(query2_s, query1_s)], 1))

                h_v1_s = self.ConvGRU(attention1_s, exemplar_s)
                h_v2_s = self.ConvGRU(attention2_s, query_s)
                h_v3_s = self.ConvGRU(attention3_s, query1_s)
                h_v4_s = self.ConvGRU(attention4_s, query2_s)

                exemplar_s = h_v1_s.clone()
                query_s = h_v2_s.clone()
                query1_s = h_v3_s.clone()
                query2_s = h_v4_s.clone()

                attention1_t = self.conv_fusion(torch.cat([self.generate_attention(exemplar_t, query_t),
                                                           self.generate_attention(exemplar_t, query1_t),
                                                           self.generate_attention(exemplar_t, query2_t)],
                                                          1))  # message passing with concat operation
                attention2_t = self.conv_fusion(torch.cat([self.generate_attention(query_t, exemplar_t),
                                                           self.generate_attention(query_t, query1_t),
                                                           self.generate_attention(query_t, query2_t)], 1))

                attention3_t = self.conv_fusion(torch.cat([self.generate_attention(query1_t, exemplar_t),
                                                           self.generate_attention(query1_t, query_t),
                                                           self.generate_attention(query1_t, query2_t)], 1))

                attention4_t = self.conv_fusion(torch.cat([self.generate_attention(query2_t, exemplar_t),
                                                           self.generate_attention(query2_t, query_t),
                                                           self.generate_attention(query2_t, query1_t)], 1))

                h_v1_t = self.ConvGRU(attention1_t, exemplar_t)
                h_v2_t = self.ConvGRU(attention2_t, query_t)
                h_v3_t = self.ConvGRU(attention3_t, query1_t)
                h_v4_t = self.ConvGRU(attention4_t, query2_t)

                exemplar_t = h_v1_t.clone()
                query_t = h_v2_t.clone()
                query1_t = h_v3_t.clone()
                query2_t = h_v4_t.clone()

                if passing_round == self.propagate_layers - 1:
                    x1s[ii, :, :, :] = self.sal_gen(h_v1, h_v1_s, h_v1_t, exemplars[ii, :, :, :][None].contiguous(), exemplars_s[ii, :, :, :][None].contiguous(), exemplars_t[ii, :, :, :][None].contiguous(),
                                                   x_size)
                    x2s[ii, :, :, :], x3s[ii, :, :, :] = self.dis_gen(h_v2, h_v2_s, h_v2_t, querys, querys_s, querys_t, h_v3, h_v3_s,h_v3_t, query1s,query1s_s,query1s_t,
                         h_v4,h_v4_s,h_v4_t,query2s,query2s_s,query2s_t, x_size)

        return x1s, x2s, x3s

    def generate_attention(self, exemplar, query):
        fea_size = query.size()[2:]

        exemplar_flat = exemplar.view(-1, self.channel, fea_size[0] * fea_size[1])  # N,C,H*W

        query_flat = query.view(-1, self.channel, fea_size[0] * fea_size[1])
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batch size x dim x num
        exemplar_corr = self.linear_e(exemplar_t)
        A = torch.bmm(exemplar_corr, query_flat)
        B = F.softmax(torch.transpose(A, 1, 2), dim=1)

        exemplar_att = torch.bmm(query_flat, B).contiguous()
        input1_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])

        input1_mask = self.gate(input1_att)
        input1_mask = self.gate_s(input1_mask)
        input1_att = input1_att * input1_mask

        return input1_att

    def sal_gen(self, input1_att, input1_att_s, input1_att_t, exemplar, exemplar_s, exemplar_t, x_size):

        input1_att = torch.cat([input1_att, exemplar], 1)
        x1 = self.sal_conv_f1(input1_att)
        x1_size = x1.size()[2:]
        input1_att_s = torch.cat([input1_att_s, exemplar_s], 1)
        input1_att_s = self.sal_conv_f1(input1_att_s)
        x1_s = F.upsample(input1_att_s, x1_size, mode='bilinear', align_corners=True)

        input1_att_t = torch.cat([input1_att_t, exemplar_t], 1)
        input1_att_t = self.sal_conv_f1(input1_att_t)
        x1_t = F.upsample(input1_att_t, x1_size, mode='bilinear', align_corners=True)

        out_c = torch.cat([x1, x1_s, x1_t], 1)
        out_c = self.sal_conv_f2(out_c)
        out_c = F.upsample(out_c, x_size, mode='bilinear', align_corners=True)
        return out_c

    def dis_gen(self, h_v2, h_v2_s, h_v2_t, querys, querys_s, querys_t, h_v3, h_v3_s,h_v3_t, query1s,query1s_s,query1s_t,
                         h_v4,h_v4_s,h_v4_t,query2s,query2s_s ,query2s_t, x_size):

        h2 = self.dis_conv1(torch.cat([h_v2, querys], 1))
        h2_size = h2.size()[2:]
        h2_s = self.dis_conv1(torch.cat([h_v2_s, querys_s], 1))
        h2_s_up = F.upsample(h2_s, h2_size, mode='bilinear', align_corners=True)
        h2_t = self.dis_conv1(torch.cat([h_v2_t, querys_t], 1))
        h2_t_up = F.upsample(h2_t, h2_size, mode='bilinear', align_corners=True)

        h2_f = torch.cat([h2, h2_s_up,h2_t_up], 1)
        h2_f = self.dis_conv1_h2(h2_f)

        h3 = self.dis_conv2(torch.cat([h_v3, query1s], 1))
        h3_size = h3.size()[2:]
        h3_s = self.dis_conv2(torch.cat([h_v3_s, query1s_s], 1))
        h3_s_up = F.upsample(h3_s, h3_size, mode='bilinear', align_corners=True)
        h3_t = self.dis_conv2(torch.cat([h_v3_t, query1s_t], 1))
        h3_t_up = F.upsample(h3_t, h3_size, mode='bilinear', align_corners=True)

        h3_f = torch.cat([h3, h3_s_up,h3_t_up], 1)
        h3_f = self.dis_conv2_h3(h3_f)

        h4 = self.dis_conv3(torch.cat([h_v4, query2s], 1))
        h4_size = h4.size()[2:]
        h4_s = self.dis_conv3(torch.cat([h_v4_s, query2s_s], 1))
        h4_s_up = F.upsample(h4_s, h4_size, mode='bilinear', align_corners=True)
        h4_t = self.dis_conv3(torch.cat([h_v4_t, query2s_t], 1))
        h4_5_up = F.upsample(h4_t, h4_size, mode='bilinear', align_corners=True)

        h4_f = torch.cat([h4, h4_s_up,h4_5_up], 1)
        h4_f = self.dis_conv3_h4(h4_f)

        H_final = torch.cat([h2_f, h3_f, h4_f], 1)
        H_final = self.dis_conv4_f(H_final)

        H_sod = self.dis_sod(H_final)
        H_dis = self.dis_dis(H_final)

        x2 = F.upsample(H_sod, x_size, mode='bilinear', align_corners=True)
        x3 = F.upsample(H_dis, x_size, mode='bilinear', align_corners=True)

        return x2, x3

# extra part
def extra_layer(base_model_cfg, resnet):
    if base_model_cfg == 'resnet':
        config = config_resnet
    merge1_layers = GeoModel()

    return resnet, merge1_layers

# TUN network
class TUN_bone(nn.Module):
    def __init__(self, base_model_cfg, base, merge1_layers):
        super(TUN_bone, self).__init__()
        self.base_model_cfg = base_model_cfg
        if self.base_model_cfg == 'resnet':
            self.convert = ConvertLayer(config_resnet['convert'])
            self.base = base
            self.merge1 = merge1_layers


    def forward(self, x1_ori, x1_mid, x1_sml, x2, x3, x4, x5, x6, x7, x8, x9, x10):
        x_size = (256, 256)
        conv2merge1_ori = self.base(x1_ori)
        conv2merge1_mid = self.base(x1_mid)
        conv2merge1_sml = self.base(x1_sml)
        conv2merge2 = self.base(x2)
        conv2merge3 = self.base(x3)
        conv2merge4 = self.base(x4)
        conv2merge5 = self.base(x5)
        conv2merge6 = self.base(x6)
        conv2merge7 = self.base(x7)
        conv2merge8 = self.base(x8)
        conv2merge9 = self.base(x9)
        conv2merge10 = self.base(x10)

        if self.base_model_cfg == 'resnet':
            conv2merge_x1_ori = self.convert(conv2merge1_ori)
            conv2merge_x1_mid = self.convert(conv2merge1_mid)
            conv2merge_x1_sml = self.convert(conv2merge1_sml)
            conv2merge_x2 = self.convert(conv2merge2)
            conv2merge_x3 = self.convert(conv2merge3)
            conv2merge_x4 = self.convert(conv2merge4)
            conv2merge_x5 = self.convert(conv2merge5)
            conv2merge_x6 = self.convert(conv2merge6)
            conv2merge_x7 = self.convert(conv2merge7)
            conv2merge_x8 = self.convert(conv2merge8)
            conv2merge_x9 = self.convert(conv2merge9)
            conv2merge_x10 = self.convert(conv2merge10)

        up_x1, up_x2, up_x3 = self.merge1(conv2merge_x1_ori,conv2merge_x1_mid, conv2merge_x1_sml, conv2merge_x2, conv2merge_x3, conv2merge_x4, \
                                          conv2merge_x5, conv2merge_x6, conv2merge_x7, conv2merge_x8, conv2merge_x9,
                                          conv2merge_x10, x_size)

        return up_x1, up_x2, up_x3


# build the whole network
def build_model(base_model_cfg='resnet'):
    if base_model_cfg == 'resnet':
        return TUN_bone(base_model_cfg, *extra_layer(base_model_cfg, resnet50()))

# weight init
def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()

