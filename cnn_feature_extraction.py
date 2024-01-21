import torch

import pandas as pd
import numpy as np

import datetime
from timeit import default_timer as timer
from tqdm.auto import tqdm

import zipfile
from pathlib import Path
import os
import shutil

import sys
sys.path.append("./util")
from dataset import VideoDataset
from model import ResNet50
# from engine import Engine

if __name__ == '__main__':
    print("="*50)

    DATABASE = "CSCVQ"
    CNN_MODULE = "ResNet50"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FRAME_BATCH_SIZE = 10

    SOURCE_DIR = Path(f"source/{DATABASE}/")
    SOURCE_VIDEO_DIR = Path(f"source/{DATABASE}/videos/")
    SOURCE_VIDEO_MOS_FILE = Path(f"source/{DATABASE}/CSCVQ1.0-MOS.xlsx")

    DATA_DIR = Path(f"data/{DATABASE}/")
    DATA_VIDEO_DIR = Path(f"data/{DATABASE}/videos/")
    DATA_VIDEO_MOS_FILE = Path(f"data/{DATABASE}/CSCVQ1.0-MOS.xlsx")

    FEATURE_DIR = Path(f"feature/{DATABASE}/{CNN_MODULE}/")

    print(f"database: {DATABASE}, CNN module: {CNN_MODULE}, device: {DEVICE}, frame_batch_size: {FRAME_BATCH_SIZE}")

    if not os.path.exists(FEATURE_DIR):
        FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(DATA_DIR) and not os.path.exists(SOURCE_DIR):
        print(f"No source found to be extracted in {SOURCE_DIR}")
        sys.exit()

    # ==================================================
    # 1. Extract video data and score data
    # ==================================================
    # Video
    print("="*50)
    if os.path.exists(DATA_VIDEO_DIR):
        print(f"Video data exists in {DATA_VIDEO_DIR}/")
    else:
        DATA_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        for dir_path, dir_names, file_names in os.walk(SOURCE_VIDEO_DIR):
            for file_name in file_names:
                file_path = os.path.join(dir_path, file_name)
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    print(f"Unzipping {DATABASE}: {file_path}")
                    zip_ref.extractall(DATA_VIDEO_DIR)

    # MOS
    if os.path.exists(DATA_VIDEO_MOS_FILE):
        print(f"MOS data exists in {DATA_VIDEO_MOS_FILE}")
    else:
        print(f"Copying {DATABASE} MOS: {SOURCE_VIDEO_MOS_FILE}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE_VIDEO_MOS_FILE, DATA_VIDEO_MOS_FILE)
    
    print(f"{DATABASE} data in: {DATA_DIR}")
    # ==================================================
    # 2. Prepare extraction data
    # ==================================================
    print("="*50)
    dataset_df = pd.read_excel(str(DATA_VIDEO_MOS_FILE),
                               header=None)
    dataset_df = dataset_df[:-1]
    # Delete last row that contains invalid label

    dataset = VideoDataset(video_dir=str(DATA_VIDEO_DIR),
                           height=720,
                           width=1280,
                           dataset_df=dataset_df)
    print(f"Number of video data to be extracted: {len(dataset)}")
    # ==================================================
    # 3. Extract CNN features
    # ==================================================
    print("="*50)

    start_time = timer()

    resnet50 = ResNet50().to(device=DEVICE)
    resnet50.eval()
    with torch.inference_mode():
        for i in tqdm(range(len(dataset))):
            video, mos = dataset[i]
            video_name = dataset.get_video_name(i)

            current = 0
            end_frame = len(video)

            feature_mean = torch.Tensor().to(device=DEVICE)
            feature_std = torch.Tensor().to(device=DEVICE)
            cnn_feature = torch.Tensor().to(device=DEVICE)

            mos = torch.tensor(mos).to(device=DEVICE).unsqueeze(dim=0)

            while current < end_frame:
                head = current
                tail = (head + FRAME_BATCH_SIZE) if (head + FRAME_BATCH_SIZE < end_frame) else end_frame
                print(f"Extracting {video_name}: index[{i}] | frames[{head}, {tail-1}]")
                batch_frames = video[head:tail]

                mean, std = resnet50(batch_frames)
                feature_mean = torch.cat((feature_mean, mean), 0)
                feature_std = torch.cat((feature_std, std), 0)

                current += FRAME_BATCH_SIZE

            cnn_feature = torch.cat((feature_mean, feature_std), 1).squeeze().numpy()
            mos = mos.numpy()


            OUTPUT_DIR = Path(f"feature/{DATABASE}/{CNN_MODULE}/{video_name}")
            if not os.path.exists(OUTPUT_DIR):
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                np.save(f"{OUTPUT_DIR}/feature", cnn_feature)
                np.save(f"{OUTPUT_DIR}/mos", mos)
    end_time = timer()
    print(f"Total extraction time: {datetime.timedelta(seconds=int(end_time-start_time))} (Hour:Minute:Second)")