import torch
import torch.nn as nn
from torch import Tensor

from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.weights = ResNet50_Weights.IMAGENET1K_V2
        # self.transforms = self.weights.transforms(antialias=True)
        # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
        # The images are resized to resize_size=[232] using interpolation=InterpolationMode.BILINEAR,
        # followed by a central crop of crop_size=[224].
        # Finally the values are first rescaled to [0.0, 1.0]
        # and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].

        self.transforms = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.module_list = nn.Sequential(
            *list(resnet50(weights=self.weights).children())[:-2]
        )

        for p in self.module_list.parameters():
            p.requires_grad = False

    def global_std_pool2d(self, x: Tensor):
        return torch.std(
            input=x.view(x.size()[0], x.size()[1], -1, 1), dim=2, keepdim=True
        )

    def forward(self, x):
        x = self.transforms(x)

        for i, module in enumerate(self.module_list):
            x = module(x)
            if i == (len(self.module_list) - 1):
                mean = nn.functional.adaptive_avg_pool2d(x, 1)
                std = self.global_std_pool2d(x)
                return mean, std


class SCVQA(nn.Module):
    def __init__(self, input_size=64, hidden_size=16, num_layers=3, device="") -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

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

        x = self.fc1(x)

        return x
