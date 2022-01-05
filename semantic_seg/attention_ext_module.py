#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedMLP(nn.Sequential):
    def __init__(
            self,
            args: List[int],
            *,
            bn: bool = False,
            activation=nn.ReLU(inplace=True),
            preact: bool = False,
            first: bool = False,
            name: str = ""
    ):
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0)) else None,
                    preact=preact
                )
            )

            
class AttentionEXTModule(nn.Module):
    def __init__(self, sum_or_maxpool: str='sum', if_softmax: bool=True, out_c: int=13):
        super().__init__()
        self.sum_or_maxpool = sum_or_maxpool # [sum | maxpool]
        self.if_softmax = if_softmax

        self.convs1 = nn.Sequential(
            nn.Conv1d(352, 512, 1), nn.BatchNorm1d(512), nn.ReLU(True),
            nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(True))

        self.convs2 = nn.Sequential(
            nn.Conv1d(352, 512, 1), nn.BatchNorm1d(512), nn.ReLU(True),
            nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(True))

        self.wei_conv = pt_util.SharedMLP([256, 32, 64, 256], bn=True)

        self.fc = nn.Linear(256, out_c)

            

    def forward(self, ext_fea: torch.Tensor = None, lab_fea: torch.Tensor = None) -> (torch.Tensor):
        # ext_fea: extension points features: N*C
        # lab_fea: labled points features: M*C
        A = ext_fea.permute(1, 0).contiguous().unsqueeze(0)
        B = lab_fea.permute(1, 0).contiguous().unsqueeze(0)

        A = self.convs1(A)
        B = self.convs2(B)

        fea_diff = A.unsqueeze(-1) - B.unsqueeze(2) 
        wei_fea = self.wei_conv(fea_diff)
        if self.if_softmax:
            wei_fea = F.softmax(wei_fea, dim=-1)

        att_fea = wei_fea * fea_diff
        if self.sum_or_maxpool == 'sum':
            att_fea = torch.sum(att_fea, dim=-1)
        elif self.sum_or_maxpool == 'maxpool':
            att_fea = F.max_pool2d(att_fea, kernel_size=[1, att_fea.size(3)])

        att_fea = torch.squeeze(att_fea)
        att_fea = att_fea.permute(1, 0).contiguous()

        fea_out = self.fc(att_fea)


        return fea_out