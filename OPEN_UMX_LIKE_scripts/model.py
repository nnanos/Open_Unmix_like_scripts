from typing import Optional, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter



class UNet(nn.Module):
    # def __init__(self,input_mean,input_scale):
    def __init__(self,):
        super().__init__()

        # Define the network components
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True)
        )
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv1_BAD = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv2 = nn.ConvTranspose2d(512, 128, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv2_BAD = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv3_BAD = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv4 = nn.ConvTranspose2d(128, 32, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv4_BAD = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv5 = nn.ConvTranspose2d(64, 16, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv5_BAD = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv6 = nn.ConvTranspose2d(32, 1, kernel_size = (5, 5), stride=(2, 2), padding=2)


        # if input_mean is not None:
        #     input_mean = torch.from_numpy(-input_mean[: 512]).float()
        # else:
        #     input_mean = torch.zeros(512)

        # if input_scale is not None:
        #     input_scale = torch.from_numpy(1.0 / input_scale[: 512]).float()
        # else:
        #     input_scale = torch.ones(512)

        # self.input_mean = Parameter(input_mean)
        # self.input_scale = Parameter(input_scale)

        # self.output_scale = Parameter(torch.ones(512).float())
        # self.output_mean = Parameter(torch.ones(512).float())


        # # Define loss list
        # self.loss_list_vocal = []
        # self.Loss_list_vocal = []

        # # Define the criterion and optimizer
        # self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        # self.crit  = nn.L1Loss()
        # self.to('cuda')



    def forward(self, x):
        """
            Generate the mask for the given mixture audio spectrogram

            Arg:    mix     (torch.Tensor)  - The mixture spectrogram which size is (B, 1, 512, 128)
            Ret:    The soft mask which size is (B, 1, 512, 128)
        """

        mix = x.detach().clone()

        # # shift and scale input to mean=0 std=1 (across all bins)
        # x = x + self.input_mean
        # x = x * self.input_scale

        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv6_out = self.conv6(conv5_out)
        deconv1_out = self.deconv1(conv6_out, output_size = conv5_out.size())
        deconv1_out = self.deconv1_BAD(deconv1_out)
        deconv2_out = self.deconv2(torch.cat([deconv1_out, conv5_out], 1), output_size = conv4_out.size())
        deconv2_out = self.deconv2_BAD(deconv2_out)
        deconv3_out = self.deconv3(torch.cat([deconv2_out, conv4_out], 1), output_size = conv3_out.size())
        deconv3_out = self.deconv3_BAD(deconv3_out)
        deconv4_out = self.deconv4(torch.cat([deconv3_out, conv3_out], 1), output_size = conv2_out.size())
        deconv4_out = self.deconv4_BAD(deconv4_out)
        deconv5_out = self.deconv5(torch.cat([deconv4_out, conv2_out], 1), output_size = conv1_out.size())
        deconv5_out = self.deconv5_BAD(deconv5_out)
        deconv6_out = self.deconv6(torch.cat([deconv5_out, conv1_out], 1), output_size = mix.size())
        

        # apply output scaling
        # deconv6_out *= self.output_scale
        # deconv6_out += self.output_mean        


        mask = F.sigmoid(deconv6_out)



        out = mask * mix

        return out