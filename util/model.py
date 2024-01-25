import torch
import torch.nn as nn
from torch import Tensor

from torchvision.models import resnet50


class ResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.module_list = nn.Sequential(*list(resnet50().children())[:-2])

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


class SCVQA(nn.Module):
    def __init__(
        self,
        device,
        input_size: int = 16,
        hidden_size: int = 64,
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
            nn.Linear(in_features=256, out_features=64),
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=4848, out_features=4096),
            nn.Linear(in_features=4096, out_features=1024),
            nn.Linear(in_features=1024, out_features=256),
            nn.Linear(in_features=256, out_features=64),
            nn.Linear(in_features=64, out_features=1),
        )

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
        # update 303 to 300 :cut 3 frames
        # [batch, 303, 16]

        # 128

        # [batch, 303, 1]

        # mean
        # batch * [1]

        x = self.fc1(x)

        return x
