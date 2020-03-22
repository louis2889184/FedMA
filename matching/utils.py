import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.optimize import linear_sum_assignment
from lapsolver import solve_dense
import time

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class SimpleCNNContainerConvBlocks(nn.Module):
    def __init__(self, input_channel, num_filters, kernel_size, output_dim=10):
        super(SimpleCNNContainerConvBlocks, self).__init__()
        '''
        A testing cnn container, which allows initializing a CNN with given dims
        We use this one to estimate matched output of conv blocks

        num_filters (list) :: number of convolution filters
        hidden_dims (list) :: number of neurons in hidden layers

        Assumptions:
        i) we use only two conv layers and three hidden layers (including the output layer)
        ii) kernel size in the two conv layers are identical
        '''
        self.conv1 = nn.Conv2d(input_channel, num_filters[0], kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x


class ModerateCNNContainerConvBlocks(nn.Module):
    def __init__(self, num_filters, output_dim=10):
        super(ModerateCNNContainerConvBlocks, self).__init__()

        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=num_filters[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=num_filters[3], out_channels=num_filters[4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[4], out_channels=num_filters[5], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        return x


class ModerateCNNContainerConvBlocksMNIST(nn.Module):
    def __init__(self, num_filters, output_dim=10):
        super(ModerateCNNContainerConvBlocksMNIST, self).__init__()

        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=num_filters[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=num_filters[3], out_channels=num_filters[4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[4], out_channels=num_filters[5], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            #print("in_channels: {}, v: {}".format(in_channels, v))
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# class VGGContainer_d(nn.Module):
#     '''
#     VGG model 
#     '''
#     def __init__(self, features, input_dim, hidden_dims, num_classes=10):
#         super(VGGContainer_d, self).__init__()
#         self.features = features
#         # note: we hard coded here a bit by assuming we only have two hidden layers
#         # self.classifier = nn.Sequential(
#         #     nn.Dropout(),
#         #     nn.Linear(input_dim, hidden_dims[0]),
#         #     nn.ReLU(True),
#         #     nn.Dropout(),
#         #     nn.Linear(hidden_dims[0], hidden_dims[1]),
#         #     nn.ReLU(True),
#         #     nn.Linear(hidden_dims[1], num_classes),
#         # )
#          # Initialize weights
#         # for m in self.modules():
#         #     if isinstance(m, nn.Conv2d):
#         #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#         #         m.weight.data.normal_(0, math.sqrt(2. / n))
#         #         m.bias.data.zero_()


#     def forward(self, x):
#         x = self.features(x)
#         # x = x.view(x.size(0), -1)
#         # x = self.classifier(x)
#         return x


def matched_vgg11_d(matched_shapes):
    # [(67, 27), (67,), (132, 603), (132,), (260, 1188), (260,), (261, 2340), (261,), (516, 2349), (516,), (517, 4644), (517,), 
    # (516, 4653), (516,), (516, 4644), (516,), (516, 515), (515,), (515, 515), (515,), (515, 10), (10,)]
    processed_matched_shape = [matched_shapes[0], 
                                'M', 
                                matched_shapes[1], 
                                'M', 
                                matched_shapes[2], 
                                matched_shapes[3], 
                                'M', 
                                matched_shapes[4], 
                                matched_shapes[5], 
                                'M', 
                                matched_shapes[6], 
                                matched_shapes[7], 
                                'M']
    return make_layers(processed_matched_shape)