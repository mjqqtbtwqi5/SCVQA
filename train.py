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
from argparse import ArgumentParser

import sys

sys.path.append("./util")
from dataset import FeatureDataset
from model import LSTM, Transformer
from engine import Engine


def mos_normalization(feature_data_list: list, mos_max: float, mos_min: float):
    for i in range(len(feature_data_list)):
        data_tup = feature_data_list[i]
        data_list = list(data_tup)
        mos = data_list[1]
        mos = np.float32((mos - mos_min) / (mos_max - mos_min))  # normalization
        data_list[1] = mos
        feature_data_list[i] = tuple(data_list)


def get_mos_max_min(feature_data_list: list):
    mos_list = [data[1] for data in feature_data_list]
    return max(mos_list), min(mos_list)


if __name__ == "__main__":
    print("=" * 50)

    _LSTM = "LSTM"
    _TRANSFORMER = "Transformer"
    MODELS = [_LSTM, _TRANSFORMER]

    _CSCVQ = "CSCVQ"
    _SCVD = "SCVD"
    DATABASES = [_CSCVQ, _SCVD]

    _ResNet50 = "ResNet50"
    CNN_EXTRACTIONS = [_ResNet50]

    parser = ArgumentParser(description="Screen Content Video Quality Assessment")
    parser.add_argument(
        "--model",
        type=str,
        choices=MODELS,
        required=True,
    )
    parser.add_argument(
        "--database",
        type=str,
        choices=DATABASES,
        required=True,
    )
    parser.add_argument(
        "--cnn_extraction",
        type=str,
        choices=CNN_EXTRACTIONS,
        required=True,
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=int, default=0.00001)
    parser.add_argument("--seed", type=int, default=22035001)

    args = parser.parse_args()

    MODEL = args.model
    DATABASE = args.database
    CNN_EXTRACTION = args.cnn_extraction
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    SEED = args.seed

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    info = {
        "DATE_TIME": None,
        "TOTAL_TIME": None,
        "DIR": None,
        "LOSS_VAL_CRITERION": None,
        "RMSE_VAL_CRITERION": None,
        "PLCC_VAL_CRITERION": None,
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

    feature_data_list = list()
    MOS_MAX, MOS_MIN = None, None

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
            # mos | float

            feature_data_list.append((feature, mos))

        MOS_MAX, MOS_MIN = get_mos_max_min(feature_data_list=feature_data_list)
        mos_normalization(
            feature_data_list=feature_data_list, mos_max=MOS_MAX, mos_min=MOS_MIN
        )

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
    info["PLCC_VAL_CRITERION"] = model_results["test_PLCC"][-1]
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
