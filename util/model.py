import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from torchvision.models import resnet50, ResNet50_Weights

import numpy as np


class ResNet50(nn.Module):
    def __init__(self) -> None:
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


class VQA_LSTM(nn.Module):
    def __init__(
        self,
        device,
        input_size: int = 64,
        hidden_size: int = 32,
        num_layers: int = 8,
    ) -> None:
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

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=16),
            nn.Linear(in_features=16, out_features=1),
        )

        # self.fc1 = nn.Linear(in_features=hidden_size, out_features=1)

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

        x = self.fc0(x)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
            device=self.device
        )
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
            device=self.device
        )

        x, _ = self.lstm(x, (h0, c0))
        # [batch, 300, 32]

        x = self.fc1(x)
        # print(x)
        # [batch, 300, 1]

        scores = torch.zeros(batch_size).to(device=self.device)
        for i in range(batch_size):
            video_batch = x[i]
            # print(video_batch.shape)
            frames_score = self.TP(video_batch)
            m = torch.mean(frames_score).to(device=self.device)
            # print(f"m: {m}")
            scores[i] = m
        return scores
