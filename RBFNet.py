
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 23:05:23 2020

@author: zhang
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from nets import edge_conv2d64
from nets import edge_conv2d128
from nets import edge_conv2d256
from nets.seaattenation import  La_Attention

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = residual + out
        out = self.relu(out)
        return out

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

def downsample():
    return nn.MaxPool2d(kernel_size=2, stride=2)

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class GCN1(nn.Module):
    def __init__(self, num_state, num_node):
        super(GCN1, self).__init__()
        self.num_state = num_state
        self.num_node = num_node
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1)

    def forward(self, seg, aj):
        n, c, h, w = seg.size()
        seg = seg.view(n, self.num_state, -1).contiguous()
        seg_similar = torch.bmm(seg, aj)
        out = self.relu(self.conv2(seg_similar))
        output = out + seg

        return output


class ISCPGCN(nn.Module):
    def __init__(self, num_in, plane_mid, mids, normalize=False):
        super(ISCPGCN, self).__init__()
        self.num_in = num_in
        self.mids = mids
        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.maxpool_c = nn.AdaptiveAvgPool2d(output_size=(1))
        self.conv_s1 = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_s11 = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_s2 = nn.Conv2d(1, 1, kernel_size=1)
        self.conv_s3 = nn.Conv2d(1, 1, kernel_size=1)
        self.mlp = nn.Linear(num_in, self.num_s)
        self.fc = nn.Conv2d(num_in, self.num_s, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.downsample = nn.AdaptiveAvgPool2d(output_size=(mids, mids))
        self.gcn = GCN1(num_state=num_in, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1)
        self.blocker = nn.BatchNorm2d(num_in)

    def forward(self, seg_ori, edge_ori):
        seg = seg_ori
        edge = edge_ori
        n, c, h, w = seg.size()

        seg_s = self.conv_s1(seg)
        theta_T = seg_s.view(n, self.num_s, -1).contiguous()
        theta = seg_s.view(n, -1, self.num_s).contiguous()
        channel_att = torch.relu(self.mlp(self.maxpool_c(seg).squeeze(3).squeeze(2))).view(n, self.num_s, -1)
        diag_channel_att = torch.bmm(channel_att, channel_att.view(n, -1, self.num_s))

        similarity_c = torch.bmm(theta, diag_channel_att)
        similarity_c = self.softmax(torch.bmm(similarity_c, theta_T))

        seg_c = self.conv_s11(seg)
        sigma = seg_c.view(n, self.num_s, -1).contiguous()
        sigma_T = seg_c.view(n, -1, self.num_s).contiguous()
        sigma_out = torch.bmm(sigma_T, sigma)

        edge_m = seg * edge

        maxpool_s, _ = torch.max(seg, dim=1)
        edge_m_pool, _ = torch.max(edge_m, dim=1)

        seg_ss = self.conv_s2(maxpool_s.unsqueeze(1)).view(n, 1, -1)
        edge_mm = self.conv_s3(edge_m_pool.unsqueeze(1)).view(n, -1, 1)

        diag_spatial_att = torch.bmm(edge_mm, seg_ss) * sigma_out
        similarity_s = self.softmax(diag_spatial_att)
        similarity = similarity_c + similarity_s

        iscp_gcn  = self.gcn(seg, similarity).view(n, self.num_in, self.mids, self.mids)


        return iscp_gcn

class GCN(nn.Module):
    def __init__(self, num_in, plane_mid, mids):
        super(GCN, self).__init__()

        self.igcn = ISCPGCN(num_in, plane_mid, mids)
        self.rnn = torch.nn.GRU(input_size=256, hidden_size=256, num_layers=1)

    def forward(self, seg, edge):
        _, c, h, w = seg.size()
        # ------------t0-------#
        updated_seg = self.gcn(seg, edge)
        updated_seg = updated_seg.view(c, -1, h * w)
        output_0, h_0 = self.rnn(updated_seg)
        # ------------t1-------#
        output = output_0.view(-1, c, h, w)
        updated_seg = self.gcn(output, edge)
        updated_seg = updated_seg.view(c, -1, h * w)
        output_1, h_1 = self.rnn(updated_seg, h_0)
        # -------------t2----------#
        output = output_1.view(-1, c, h, w)
        updated_seg = self.gcn(output, edge)
        updated_seg = updated_seg.view(c, -1, h * w)
        output_2, h_2 = self.rnn(updated_seg, h_1)

        # reshape back
        output_2 = output_2.view(-1, c, h, w)

        return output_2




class EEblock(nn.Module):
    def __init__(self, channel):
        super(EEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.sconv13 = nn.Conv2d(channel ,channel, kernel_size=(1 ,3), padding=(0 ,1))
        self.sconv31 = nn.Conv2d(channel ,channel, kernel_size=(3 ,1), padding=(1 ,0))

    def forward(self, y, x):
        # y = torch.nn.functional.relu(self.adj)
        b, c, H, W = x.size()

        x1 = self.sconv13(x)
        x2 = self.sconv31(x)

        y1 = self.sconv13(y)
        y2 = self.sconv31(y)

        map_y13 = torch.sigmoid(self.avg_pool(y1).view(b,c,1,1))
        map_y31 = torch.sigmoid(self.avg_pool(y2).view(b,c,1 ,1))

        k = x1 * map_y31 + x2 * map_y13 + x

        return k

class RBFNet(nn.Module):
    def __init__(self,  n_classes):
        super(RBFNet, self).__init__()
        self.n_classes = n_classes
        self.down = downsample()
        self.adj = torch.nn.Parameter(torch.ones((1,3,512, 512), dtype=torch.float32))
        self.Conv1 = conv_block(ch_in=3, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)

        self.Conv5 = conv_block(ch_in=3, ch_out=64)
        self.Conv6 = conv_block(ch_in=64, ch_out=128)
        self.Conv7 = conv_block(ch_in=128, ch_out=256)
        self.Conv8 = conv_block(ch_in=256, ch_out=512)
        self.GCN_layer = GCN(num_in=2, plane_mid=1, mids=32)
        self.attn1 = La_Attention(dim1, key_dim=key_dim1, num_heads=num_heads, attn_ratio=attn_ratio,
                                  activation=act_layer, )
        self.attn2 = La_Attention(dim2, key_dim=key_dim2, num_heads=num_heads, attn_ratio=attn_ratio,
                                   activation=act_layer, )
        self.attn3 = La_Attention(dim3, key_dim=key_dim3, num_heads=num_heads, attn_ratio=attn_ratio,
                                   activation=act_layer, )
        self.attn4 = La_Attention(dim4, key_dim=key_dim4, num_heads=num_heads, attn_ratio=attn_ratio,
                                   activation=act_layer, )

        self.EEblock1 = EEblock(channel=256)
        self.EEblock2 = EEblock(channel=128)
        self.EEblock3 = EEblock(channel=64)

        self.Up4 = up_conv(512 ,256)
        self.Up_conv4 = Decoder(512, 256)

        self.Up3 = up_conv(256 ,128)
        self.Up_conv3 = Decoder(256, 128)

        self.Up2 = up_conv(128 ,64)
        self.Up_conv2 = Decoder(128, 64)

        self.fconv = nn.Conv2d(64 ,2, kernel_size=1, padding=0)

    def forward(self, x):
        y = torch.nn.functional.relu(self.adj)
        x1 = self.Conv1(x)
        x9 = self.attn1(x1)
        x2 = self.down(x9)

        x3 = self.Conv2(x2)
        x10 = self.attn1(x3)
        x4 = self.down(x10)

        x5 = self.Conv3(x4)
        x11 = self.attn1(x5)
        x6 = self.down(x11)

        x7 = self.Conv4(x6)
        x12 = self.attn1(x7)
        x8 = self.down(x12)

        e1 = edge_conv2d64(x1)
        e2 = edge_conv2d128(x2)
        e3 = edge_conv2d256(x3)

        y1 = self.Conv5(y) +e1

        y2 = self.down(y1)
        y2 = self.Conv6(y2) +e2

        y3 = self.down(y2)
        y3 = self.Conv7(y3) +e3

        y4 = self.down(y3)
        y4 = self.Conv8(y4)

        GCN_output = self.GCN_layer(y4)



        m3 = self.EEblock1(GCN_output,x8)
        d4 = self.Up_conv4(m3)
        d4 = self.Up4(d4)


        m2 = self.EEblock2(d4,x6)
        d3 = self.Up_conv3(m2)
        d3 = self.Up3(d3)

        m1 = self.EEblock3(d3,x4)
        d2 = self.Up_conv2(m1)
        d2 = self.Up2(d2)



        out = self.fconv(d2)

        return out



