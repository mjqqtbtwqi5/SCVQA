import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

from tqdm.auto import tqdm

import math


class Engine:
    def __init__(
        self, device: str, epochs: int, mos_max: float, mos_min: float
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.mos_max = mos_max
        self.mos_min = mos_min

    def up_scale(self, x):
        x = x * (self.mos_max - self.mos_min) + self.mos_min
        return x

    def train_step(
        self,
        model: Module,
        loss_fn: Module,
        optimizer: Optimizer,
        dataloader: DataLoader,
    ):
        model.train()

        train_loss, train_RMSE, train_PLCC, train_SROCC = 0, 0, 0, 0

        y_list, y_pred_list = list(), list()

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device=self.device), y.to(device=self.device)

            # 1. Prediction
            y_pred = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            # 6. Saving for metrics calculation
            batch_size = len(y)
            for i in range(batch_size):
                _y = y[i].item()
                _y_pred = y_pred[i].item()
                y_list.append(_y)
                y_pred_list.append(_y_pred)

            print(
                f"Training batch[{batch}]: last record -> y: {self.up_scale(y_list[-1])} | y_pred: {self.up_scale(y_pred_list[-1])}"
            )  # print y and y pred values of the last one

        train_loss = train_loss / len(dataloader)
        train_RMSE = math.sqrt(mean_squared_error(y_list, y_pred_list))
        train_PLCC = pearsonr(y_pred_list, y_list)[0]
        train_SROCC = spearmanr(y_pred_list, y_list)[0]

        return train_loss, train_RMSE, train_PLCC, train_SROCC

    def test_step(self, model: Module, dataloader: DataLoader, loss_fn: Module):
        test_loss, test_RMSE, test_PLCC, test_SROCC = 0, 0, 0, 0

        y_list, test_y_pred_list = list(), list()

        model.eval()
        with torch.inference_mode():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device=self.device), y.to(device=self.device)

                # 1. Forward pass
                test_y_pred = model(X)

                # 2. Calculate and accumulate loss
                loss = loss_fn(test_y_pred, y)
                test_loss += loss.item()

                # 3. Saving for metrics calculation
                batch_size = len(y)
                for i in range(batch_size):
                    _y = y[i].item()
                    _test_y_pred = test_y_pred[i].item()
                    y_list.append(_y)
                    test_y_pred_list.append(_test_y_pred)
                print(
                    f"Testing  batch[{batch}]: last record -> y: {self.up_scale(y_list[-1])} | y_pred: {self.up_scale(test_y_pred_list[-1])}"
                )  # print y and y pred values of the last one

        test_loss = test_loss / len(dataloader)
        test_RMSE = math.sqrt(mean_squared_error(y_list, test_y_pred_list))
        test_PLCC = pearsonr(test_y_pred_list, y_list)[0]
        test_SROCC = spearmanr(test_y_pred_list, y_list)[0]

        return test_loss, test_RMSE, test_PLCC, test_SROCC

    def train(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ):
        results = {
            f"train_{type(loss_fn).__name__}": [],
            "train_RMSE": [],
            "train_PLCC": [],
            "train_SROCC": [],
            f"test_{type(loss_fn).__name__}": [],
            "test_RMSE": [],
            "test_PLCC": [],
            "test_SROCC": [],
        }

        for epoch in tqdm(range(self.epochs)):
            train_loss, train_RMSE, train_PLCC, train_SROCC = self.train_step(
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                dataloader=train_dataloader,
            )

            test_loss, test_RMSE, test_PLCC, test_SROCC = self.test_step(
                model=model, dataloader=test_dataloader, loss_fn=loss_fn
            )

            print(
                f"[Training] Epoch: {epoch+1} | "
                f"{type(loss_fn).__name__}: {train_loss:.5f} | "
                f"RMSE: {train_RMSE:.5f} | "
                f"PLCC: {train_PLCC:.5f} | "
                f"SROCC: {train_SROCC:.5f}"
            )

            print(
                f"[Testing]  Epoch: {epoch+1} | "
                f"{type(loss_fn).__name__}: {test_loss:.5f} | "
                f"RMSE: {test_RMSE:.5f} | "
                f"PLCC: {test_PLCC:.5f} | "
                f"SROCC: {test_SROCC:.5f}"
            )

            results[f"train_{type(loss_fn).__name__}"].append(train_loss)
            results["train_RMSE"].append(train_RMSE)
            results["train_PLCC"].append(train_PLCC)
            results["train_SROCC"].append(train_SROCC)

            results[f"test_{type(loss_fn).__name__}"].append(test_loss)
            results["test_RMSE"].append(test_RMSE)
            results["test_PLCC"].append(test_PLCC)
            results["test_SROCC"].append(test_SROCC)

        return results
