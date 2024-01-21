import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.weights = ResNet50_Weights.IMAGENET1K_V2
        self.transforms = self.weights.transforms(antialias=True)
        # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
        # The images are resized to resize_size=[232] using interpolation=InterpolationMode.BILINEAR, 
        # followed by a central crop of crop_size=[224]. 
        # Finally the values are first rescaled to [0.0, 1.0] 
        # and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].

        self.module_list = nn.Sequential(
            *list(resnet50(weights=self.weights).children())[:-2]
        )

        for p in self.module_list.parameters():
            p.requires_grad = False
    
    def global_std_pool2d(self, x:Tensor):
        return torch.std(input=x.view(x.size()[0], x.size()[1], -1, 1),
                         dim=2, keepdim=True)

    def forward(self, x):
        for frame in x:
            frame = self.transforms(frame)

        for i, module in enumerate(self.module_list):
            x = module(x)
            if i == (len(self.module_list) - 1):
                mean = nn.functional.adaptive_avg_pool2d(x, 1)
                std = self.global_std_pool2d(x)
                return mean, std

class SCVQA(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # self.regressor = nn.Sequential(
        #     nn.Linear(in_features=4096,
        #               out_features=128),
        #     nn.LSTM(input_size=128,
        #             hidden_size=128),
        #     nn.Linear(in_features=128,
        #               out_features=1),
        # )
        self.fc0 = nn.Linear(in_features=4096,
                             out_features=128)
        
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=32)
        
        self.fl = nn.Flatten()
        
        self.fc1 = nn.Linear(in_features=9600,
                             out_features=1)

    def forward(self, x):
        print(f"x: {x.shape}")
        x = self.fc0(x)
        print(f"self.fc0(x): {x.shape}")
        x, _ = self.lstm(x)
        print(f"self.lstm(x): {x.shape}")
        x = self.fl(x)
        print(f"self.fl(x): {x.shape}")
        x = self.fc1(x)
        print(f"self.fc1(x): {x.shape}")

        return x