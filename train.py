import torch
from torch import nn
from torch.utils.data import DataLoader

import os
import numpy as np
import pandas as pd
from pathlib import Path
import datetime
from timeit import default_timer as timer

import sys

sys.path.append("./util")
from dataset import FeatureDataset
from model import VQA_LSTM
from engine import Engine

if __name__ == "__main__":
    print("=" * 50)

    DATABASE = "CSCVQ"
    CNN_MODULE = "ResNet50"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8
    # NUM_WORKERS = os.cpu_count()
    NUM_WORKERS = 0
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    SEED = 22035001
    RNN = ["LSTM", "Transformer"]
    MODEL_IMPL = RNN[0]
    # MODEL_IMP = RNN[1]
    USE_TRAINED = True  # load trained model

    info = {
        "DATABASE": DATABASE,
        "TRAIN_DATA_SIZE": None,
        "TEST_DATA_SIZE": None,
        "CNN_MODULE": CNN_MODULE,
        "DEVICE": DEVICE,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "NUM_EPOCHS": NUM_EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "SEED": SEED,
        "MODEL_IMPL": MODEL_IMPL,
        "DATE_TIME": None,
        "TOTAL_TIME": None,
    }

    FEATURE_DIR = Path(f"feature/{DATABASE}/{CNN_MODULE}/")
    DATA_VIDEO_MOS_FILE = Path(f"data/{DATABASE}/CSCVQ1.0-MOS.xlsx")

    MODEL_DIR = Path(f"model/{MODEL_IMPL}/")
    MODEL_LATEST_DIR = Path(f"model/{MODEL_IMPL}/latest/")
    MODEL_LATEST_PT = Path(f"model/{MODEL_IMPL}/latest/model.pt")
    MODEL_LATEST_RESULT = Path(f"model/{MODEL_IMPL}/latest/result.csv")
    MODEL_LATEST_INFO = Path(f"model/{MODEL_IMPL}/latest/info.csv")

    print(
        f"database: {DATABASE}, CNN module: {CNN_MODULE}, Model implement: {MODEL_IMPL}, device: {DEVICE}, batch_size: {BATCH_SIZE}, num_workers: {NUM_WORKERS}, learning_rate: {LEARNING_RATE}, num_epochs: {NUM_EPOCHS}, seed: {SEED}"
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

    model = VQA_LSTM(device=DEVICE, input_size=64, hidden_size=32, num_layers=8).to(
        device=DEVICE
    )

    if os.path.exists(MODEL_LATEST_PT) and USE_TRAINED:
        print(f"Load model from {MODEL_LATEST_PT}")
        model.load_state_dict(torch.load(f=MODEL_LATEST_PT))

    # loss_fn = nn.L1Loss()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    engine = Engine(device=DEVICE, epochs=NUM_EPOCHS, mos_max=MOS_MAX, mos_min=MOS_MIN)

    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info["DATE_TIME"] = date_time
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
    info["TOTAL_TIME"] = total_time
    print(f"Total training & testing time: {total_time}")

    if os.path.exists(MODEL_LATEST_PT):
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        MODEL_LATEST_DIR.rename(MODEL_DIR / now)

    MODEL_LATEST_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        obj=model.state_dict(),
        f=str(MODEL_LATEST_PT),
    )

    model_results_df = pd.DataFrame(model_results)
    model_results_df.to_csv(str(MODEL_LATEST_RESULT), index=False)

    info_df = pd.DataFrame(info, index=[0])
    info_df.to_csv(str(MODEL_LATEST_INFO))
