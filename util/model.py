import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from torchvision.models import resnet50, ResNet50_Weights

import numpy as np
import math


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.module_list = nn.Sequential(
            *list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).children())[:-2]
        )

        for p in self.module_list.parameters():
            p.requires_grad = False

    def global_std_pool2d(self, x: Tensor):
        x = torch.std(x.view(x.size()[0], x.size()[1], -1, 1), dim=2, keepdim=True)
        return x

    def forward(self, x):
        for i, module in enumerate(self.module_list):
            x = module(x)
            if i == (len(self.module_list) - 1):
                mean = nn.functional.adaptive_avg_pool2d(x, 1)
                std = self.global_std_pool2d(x)
                return mean, std


class ANN(nn.Module):
    def __init__(self, input_size=4096, reduced_size=128, n_ANNlayers=1, dropout_p=0.5):
        super().__init__()
        self.n_ANNlayers = n_ANNlayers
        self.fc0 = nn.Linear(input_size, reduced_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(reduced_size, reduced_size)

    def forward(self, input):
        input = self.fc0(input)
        for i in range(self.n_ANNlayers - 1):
            input = self.fc(self.dropout(F.relu(input)))
        return input


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = F.pad(
            x, (0, 0, 1, 0, 0, 0), mode="constant"
        )  # add 0s vector as [batch_size, 300, d_model] -> [batch_size, 301, d_model]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
        self,
        device,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()

        self.device = device

        # self.fc0 = nn.Linear(4096, d_model)
        self.ann = ANN(4096, d_model, 2)

        self.pos_encoder = PositionalEncoding(d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
        )
        # encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            # encoder_norm
        )

        self.fc1 = nn.Linear(d_model, 1)

    def forward(self, x):

        # x = self.fc0(x)
        x = self.ann(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.fc1(x)
        # print(x.shape)  # [batch_size, 301, 1]

        x = x.squeeze(dim=2)
        # print(x.shape)  # [batch_size, 301]
        x = x[:, 0]  # use first reference
        # print(x.shape)  # [batch_size]
        return x


class LSTM(nn.Module):
    def __init__(
        self,
        device,
        input_size: int = 64,
        hidden_size: int = 32,
        num_layers: int = 8,
    ):
        super().__init__()

        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.fc0 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=1024),
            nn.Linear(in_features=1024, out_features=256),
            nn.Linear(in_features=256, out_features=input_size),
        )

        self.ann = ANN(4096, input_size, 1)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc1 = nn.Linear(in_features=hidden_size, out_features=1)

    def TP(self, q, tau=12, beta=0.5):
        """subjectively-inspired temporal pooling"""
        q = torch.unsqueeze(torch.t(q), 0)
        qm = -float("inf") * torch.ones((1, 1, tau - 1)).to(q.device)
        qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)  #
        l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)
        m = F.avg_pool1d(
            torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1
        )
        n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
        m = m / n
        return beta * m + (1 - beta) * l

    def forward(self, x):
        batch_size = x.size(0)

        # x = self.fc0(x)

        x = self.ann(x)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
            device=self.device
        )
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
            device=self.device
        )

        x, _ = self.lstm(x, (h0, c0))

        x = self.fc1(x)
        # x = self.fc1(x[:, -1, :]) #lstm get last only

        scores = torch.zeros(batch_size).to(device=self.device)
        for i in range(batch_size):
            video_batch = x[i]
            frames_score = self.TP(video_batch)
            m = torch.mean(frames_score).to(device=self.device)
            scores[i] = m
        return scores
