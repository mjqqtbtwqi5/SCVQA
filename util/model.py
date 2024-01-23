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
    def __init__(self) -> None:
        super().__init__()

        self.fc0 = nn.Linear(in_features=4096, out_features=128)

        self.lstm = nn.LSTM(
            input_size=128, hidden_size=64, num_layers=1, batch_first=True
        )

        self.fc = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.fc0(x)
        # h0 = torch.zeros(2, 1, 64)
        # c0 = torch.zeros(2, 1, 64)
        x, _ = self.lstm(x)
        # x, _ = self.lstm(x, (h0, c0))

        # x = self.fc(x[:, -1, :])
        x = self.fc(x)

        x = torch.mean(x)
        return x
