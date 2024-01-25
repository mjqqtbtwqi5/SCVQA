import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr


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

        train_loss, train_PCC, train_SROCC = 0, 0, 0

        y_list, y_pred_list = list(), list()

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

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
            for i in range(len(y)):
                _y = self.up_scale(y[i].item())
                _y_pred = self.up_scale(y_pred[i].item())
                # print(f"y: {_y} | y_pred: {_y_pred}")
                y_list.append(_y)
                y_pred_list.append(_y_pred)

        train_loss = train_loss / len(dataloader)
        train_PCC = pearsonr(y_pred_list, y_list)[0]
        train_SROCC = spearmanr(y_pred_list, y_list)[0]

        return train_loss, train_PCC, train_SROCC

    def test_step(self, model: Module, dataloader: DataLoader, loss_fn: Module):
        test_loss, test_PCC, test_SROCC = 0, 0, 0

        y_list, test_y_pred_list = list(), list()

        model.eval()
        with torch.inference_mode():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)

                # 1. Forward pass
                test_y_pred = model(X)

                # 2. Calculate and accumulate loss
                loss = loss_fn(test_y_pred, y)
                test_loss += loss.item()

                # 3. Saving for metrics calculation
                for i in range(len(y)):
                    _y = self.up_scale(y[i].item())
                    _test_y_pred = self.up_scale(test_y_pred[i].item())
                    print(f"y: {_y} | test_y_pred: {_test_y_pred}")
                    y_list.append(_y)
                    test_y_pred_list.append(_test_y_pred)

        test_loss = test_loss / len(dataloader)
        test_PCC = pearsonr(test_y_pred_list, y_list)[0]
        test_SROCC = spearmanr(test_y_pred_list, y_list)[0]

        return test_loss, test_PCC, test_SROCC

    def train(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ):
        results = {
            "train_loss": [],
            "train_PCC": [],
            "train_SROCC": [],
            "test_loss": [],
            "test_PCC": [],
            "test_SROCC": [],
        }

        for epoch in tqdm(range(self.epochs)):
            train_loss, train_PCC, train_SROCC = self.train_step(
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                dataloader=train_dataloader,
            )

            test_loss, test_PCC, test_SROCC = self.test_step(
                model=model, dataloader=test_dataloader, loss_fn=loss_fn
            )

            print(
                f"[Training] Epoch: {epoch+1} | "
                f"{type(loss_fn).__name__} loss: {train_loss:.4f} | "
                f"PCC: {train_PCC:.4f} | "
                f"SROCC: {train_SROCC:.4f}"
            )

            print(
                f"[Testing]  Epoch: {epoch+1} | "
                f"{type(loss_fn).__name__} loss: {test_loss:.4f} | "
                f"PCC: {test_PCC:.4f} | "
                f"SROCC: {test_SROCC:.4f}"
            )

            results["train_loss"].append(train_loss)
            results["train_PCC"].append(train_PCC)
            results["train_SROCC"].append(train_SROCC)

            results["test_loss"].append(test_loss)
            results["test_PCC"].append(test_PCC)
            results["test_SROCC"].append(test_SROCC)

        return results
