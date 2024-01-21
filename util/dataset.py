import torch
from torch import Tensor
from torch.utils.data import Dataset

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

    def __init__(self, video_dir: str, height: int, width: int, dataset_df: DataFrame) -> None:
        self.video_dir = video_dir
        self.height = height
        self.width = width
        self.dataset_df = dataset_df

    def get_video_name(self, index: int) -> str:
        row = self.dataset_df.iloc[index]
        video_name = str(row[0])
        return video_name

    def load_video_mos(self, index: int) -> Tuple[Tensor, float]:

        row = self.dataset_df.iloc[index]
        file_name, mos = str(row[0]), float(row[21])
        video_path = f"{self.video_dir}/{file_name}"

        video = skvideo.io.vread(video_path,
                                      self.height,
                                      self.width,
                                      inputdict={'-pix_fmt':'yuvj420p'})
        
        video = torch.permute(torch.from_numpy(video), (0, 3, 1, 2)).float()

        return video, mos
    
    def __len__(self) -> int:
        return len(self.dataset_df)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, float]:
        return self.load_video_mos(index)