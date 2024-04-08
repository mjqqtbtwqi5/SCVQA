import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np

import os
from pathlib import Path
import shutil


if __name__ == "__main__":
    print("=" * 50)

    _CSCVQ = "CSCVQ"
    _SCVD = "SCVD"
    DATABASE = _SCVD

    _ResNet18 = "ResNet18"
    _ResNet34 = "ResNet34"
    _ResNet50 = "ResNet50"
    _ResNet101 = "ResNet101"

    CNN_EXTRACTION_CONCATE_1 = _ResNet50
    CNN_EXTRACTION_CONCATE_2 = _ResNet34
    CNN_EXTRACTION_CONCATE = f"{CNN_EXTRACTION_CONCATE_1}_{CNN_EXTRACTION_CONCATE_2}"

    FEATURE_DIR = Path(f"feature/{DATABASE}/{CNN_EXTRACTION_CONCATE_1}/")

    video_feature_dir_list = [f.path for f in os.scandir(FEATURE_DIR) if f.is_dir()]

    for video_feature_dir in video_feature_dir_list:
        mos_file = f"{video_feature_dir}/mos.npy"

        feature_file1 = f"{video_feature_dir}/feature.npy"
        feature1 = np.load(feature_file1)
        feature1 = torch.from_numpy(feature1)

        feature_file2 = f"{video_feature_dir.replace(CNN_EXTRACTION_CONCATE_1, CNN_EXTRACTION_CONCATE_2)}/feature.npy"
        feature2 = np.load(feature_file2)
        feature2 = torch.from_numpy(feature2)

        feature_concat_file = f"{video_feature_dir.replace(CNN_EXTRACTION_CONCATE_1, CNN_EXTRACTION_CONCATE)}"
        feature_concat = torch.cat((feature1, feature2), dim=1)

        feature_concat_file = Path(feature_concat_file)
        if not os.path.exists(Path(feature_concat_file)):
            print(
                f"Concatenating {CNN_EXTRACTION_CONCATE_1} and {CNN_EXTRACTION_CONCATE_2} to {feature_concat_file}"
            )
            feature_concat_file.mkdir(parents=True, exist_ok=True)
            np.save(f"{feature_concat_file}/feature", feature_concat)
            shutil.copyfile(
                mos_file,
                f"{video_feature_dir.replace(CNN_EXTRACTION_CONCATE_1, CNN_EXTRACTION_CONCATE)}/mos.npy",
            )
