import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

import skvideo
import skvideo.io

from typing import Tuple
from pandas import DataFrame


class FeatureDataset(Dataset):
    def __init__(self, dataset: list) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Tensor, float]:
        feature = self.dataset[index][0]
        mos = self.dataset[index][1]
        return (feature, mos)


class VideoDataset(Dataset):
    def __init__(
        self,
        video_dir: str,
        height: int,
        width: int,
        dataset_df: DataFrame,
        max_frame_size: int,
        vid_idx: int,
        mos_idx: int,
    ) -> None:
        self.video_dir = video_dir
        self.height = height
        self.width = width
        self.dataset_df = dataset_df
        self.max_frame_size = max_frame_size

        self.vid_idx = vid_idx
        self.mos_idx = mos_idx

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_video_info(self, index: int) -> Tuple[str, float]:
        row = self.dataset_df.iloc[index]
        file_name, mos = str(row[self.vid_idx]), float(row[self.mos_idx])
        return file_name, mos

    def load_video_mos(self, index: int) -> Tuple[Tensor, float]:
        file_name, mos = self.get_video_info(index=index)
        video_path = f"{self.video_dir}/{file_name}"

        video = skvideo.io.vread(
            video_path, self.height, self.width, inputdict={"-pix_fmt": "yuvj420p"}
        )

        frame_size, height, width, channel = video.shape

        if frame_size > self.max_frame_size:
            frame_size = self.max_frame_size

        transformed_video = torch.zeros([frame_size, channel, height, width])

        for i in range(frame_size):
            frame = video[i]
            frame = Image.fromarray(frame)
            frame = self.transforms(frame)
            transformed_video[i] = frame
        # video = torch.permute(torch.from_numpy(video), (0, 3, 1, 2)).float()
        # frame, channel, height, width

        # torch.float32, float
        return transformed_video, mos

    def __len__(self) -> int:
        return len(self.dataset_df)

    def __getitem__(self, index: int) -> Tuple[Tensor, float]:
        return self.load_video_mos(index)
