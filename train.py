import torch
from torch import nn
from torch.utils.data import DataLoader

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
import datetime
from timeit import default_timer as timer

import sys

sys.path.append("./util")
from dataset import FeatureDataset
from model import LSTM, Transformer
from engine import Engine

if __name__ == "__main__":
    print("=" * 50)

    DATABASE = "CSCVQ"
    CNN_EXTRACTION = "ResNet50"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8
    NUM_WORKERS = 0
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    SEED = 22035001

    _LSTM = "LSTM"
    _TRANSFORMER = "Transformer"
    MODEL = _LSTM

    info = {
        "DATE_TIME": None,
        "TOTAL_TIME": None,
        "DIR": None,
        "LOSS_VAL_CRITERION": None,
        "RMSE_VAL_CRITERION": None,
        "PCC_VAL_CRITERION": None,
        "SROCC_VAL_CRITERION": None,
        "TRAIN_DATA_SIZE": None,
        "TEST_DATA_SIZE": None,
        "MODEL": MODEL,
        "DATABASE": DATABASE,
        "CNN_EXTRACTION": CNN_EXTRACTION,
        "DEVICE": DEVICE,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "NUM_EPOCHS": NUM_EPOCHS,
        "SEED": SEED,
        "LEARNING_RATE": LEARNING_RATE,
    }

    FEATURE_DIR = Path(f"feature/{DATABASE}/{CNN_EXTRACTION}/")
    DATA_VIDEO_MOS_FILE = Path(f"data/{DATABASE}/CSCVQ1.0-MOS.xlsx")

    MODEL_DIR = Path(f"model/{MODEL}/{DATABASE}/{CNN_EXTRACTION}/")
    MODEL_DIR_HIST_FILE = Path(f"model/{MODEL}/{DATABASE}/{CNN_EXTRACTION}/history.csv")

    print(f"[{MODEL}-based] | database: {DATABASE}, CNN extraction: {CNN_EXTRACTION}")

    print(
        f"device: {DEVICE}, batch_size: {BATCH_SIZE}, num_workers: {NUM_WORKERS}, num_epochs: {NUM_EPOCHS}, seed: {SEED}, learning_rate: {LEARNING_RATE}"
    )

    # ==================================================
    # 1. Data preparation
    # ==================================================
    print("=" * 50)

    dataset_df = pd.read_excel(str(DATA_VIDEO_MOS_FILE), header=None)
    dataset_df = dataset_df[:-1]
    # Delete last row that contains invalid label

    MOS_MAX = dataset_df[21].max()
    MOS_MIN = dataset_df[21].min()

    feature_data_list = list()

    if not os.path.exists(FEATURE_DIR):
        print(f"Video feature not exists in {FEATURE_DIR}/")
        sys.exit()
    else:
        video_feature_dir_list = [f.path for f in os.scandir(FEATURE_DIR) if f.is_dir()]

        for video_feature_dir in video_feature_dir_list:
            feature_file = f"{video_feature_dir}/feature.npy"
            mos_file = f"{video_feature_dir}/mos.npy"

            feature = np.load(feature_file)
            feature = torch.from_numpy(feature)
            # [frames, feature] | Tensor | torch.Size([300, 4096])

            mos = np.load(mos_file)
            mos = np.float32(mos.item())
            mos = np.float32((mos - MOS_MIN) / (MOS_MAX - MOS_MIN))  # normalization
            # mos | float

            feature_data_list.append((feature, mos))

    random.seed(SEED)
    random.shuffle(feature_data_list)

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

    info["TRAIN_DATA_SIZE"] = len(train_dataset)
    info["TEST_DATA_SIZE"] = len(test_dataset)

    print(
        f"Number of training data: {info['TRAIN_DATA_SIZE']} & testing data: {info['TEST_DATA_SIZE']}"
    )
    # ==================================================
    # 2. Training and testing step
    # ==================================================
    print("=" * 50)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    model = (
        LSTM(device=DEVICE).to(device=DEVICE)
        if MODEL == _LSTM
        else Transformer(device=DEVICE).to(device=DEVICE)
    )

    if os.path.exists(MODEL_DIR_HIST_FILE):
        hist_df = pd.read_csv(MODEL_DIR_HIST_FILE)
        model_file = Path(MODEL_DIR / hist_df["DIR"].iloc[-1] / "model.pt")
        if os.path.exists(model_file):
            print(f"Load model from {model_file}")
            model.load_state_dict(torch.load(f=str(model_file)))

    # loss_fn = nn.L1Loss()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    engine = Engine(device=DEVICE, epochs=NUM_EPOCHS, mos_max=MOS_MAX, mos_min=MOS_MIN)

    start_time = timer()
    model_results = engine.train(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
    )
    end_time = timer()
    total_time = (
        f"{datetime.timedelta(seconds=int(end_time-start_time))} (Hour:Minute:Second)"
    )
    print(f"Total training & testing time: {total_time}")

    info["TOTAL_TIME"] = total_time

    info["LOSS_VAL_CRITERION"] = model_results[f"test_{type(loss_fn).__name__}"][-1]
    info["RMSE_VAL_CRITERION"] = model_results["test_RMSE"][-1]
    info["PCC_VAL_CRITERION"] = model_results["test_PCC"][-1]
    info["SROCC_VAL_CRITERION"] = model_results["test_SROCC"][-1]

    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    info["DATE_TIME"] = date_time

    # Save model, result, history
    dir = now.strftime("%Y%m%d_%H%M%S")
    info["DIR"] = dir

    model_dir = Path(MODEL_DIR / dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # model
    model_file = model_dir / "model.pt"
    torch.save(
        obj=model.state_dict(),
        f=str(model_file),
    )

    # result
    result_file = model_dir / "result.csv"
    model_results_df = pd.DataFrame(model_results)
    model_results_df.to_csv(str(result_file), index=False)

    # history
    info_df = pd.DataFrame(info, index=[0])
    if os.path.exists(MODEL_DIR_HIST_FILE):
        info_df.to_csv(str(MODEL_DIR_HIST_FILE), mode="a", index=False, header=False)
    else:
        info_df.to_csv(str(MODEL_DIR_HIST_FILE), index=False)
