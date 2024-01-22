import torch
from torch import nn
from torch.utils.data import DataLoader

import os
import numpy as np
from pathlib import Path
import datetime
from timeit import default_timer as timer

import sys

sys.path.append("./util")
from dataset import FeatureDataset
from model import SCVQA
from engine import Engine

if __name__ == "__main__":
    print("=" * 50)

    DATABASE = "CSCVQ"
    CNN_MODULE = "ResNet50"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 1
    # NUM_WORKERS = os.cpu_count()
    NUM_WORKERS = 0
    NUM_EPOCHS = 500
    SEED = 22035001

    FEATURE_DIR = Path(f"feature/{DATABASE}/{CNN_MODULE}/")

    print(
        f"database: {DATABASE}, CNN module: {CNN_MODULE}, device: {DEVICE}, batch_size: {BATCH_SIZE}, num_workers: {NUM_WORKERS}, num_epochs: {NUM_EPOCHS}, seed: {SEED}"
    )

    # ==================================================
    # 1. Data preparation
    # ==================================================
    print("=" * 50)

    feature_data_list = list()

    if not os.path.exists(FEATURE_DIR):
        print(f"Video feature not exists in {FEATURE_DIR}/")
        sys.exit()
    else:
        count = 0
        os.scandir(FEATURE_DIR)
        video_feature_dir_list = [f.path for f in os.scandir(FEATURE_DIR) if f.is_dir()]

        for video_feature_dir in video_feature_dir_list:
            feature_file = f"{video_feature_dir}/feature.npy"
            mos_file = f"{video_feature_dir}/mos.npy"

            feature = np.load(feature_file)
            feature = torch.from_numpy(feature).to(device=DEVICE)
            # [frames, feature] | Tensor | torch.Size([300, 4096])

            mos = np.load(mos_file)
            mos = mos.item()
            # mos | float | 55.5

            feature_data_list.append((feature, mos))

    TRAIN_SPLIT = int(0.8 * len(feature_data_list))
    train_data_list = feature_data_list[:TRAIN_SPLIT]
    test_data_list = feature_data_list[TRAIN_SPLIT:]

    train_dataset = FeatureDataset(dataset=train_data_list)
    test_dataset = FeatureDataset(dataset=test_data_list)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
    )

    print(
        f"Number of training data: {len(train_dataset)} & testing data: {len(test_dataset)}"
    )
    # ==================================================
    # 2. Training and testing step
    # ==================================================
    print("=" * 50)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    model = SCVQA().to(device=DEVICE)

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    engine = Engine(device=DEVICE, epochs=NUM_EPOCHS)

    start_time = timer()
    model_results = engine.train(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
    )
    end_time = timer()
    print(
        f"Total training time: {datetime.timedelta(seconds=int(end_time-start_time))} (Hour:Minute:Second)"
    )
    # print(model_results)
    print(
        f"Total number of epochs: {NUM_EPOCHS} - Total number of results: {len(model_results['train_loss'])}"
    )
