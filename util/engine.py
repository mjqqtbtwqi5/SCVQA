import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm.auto import tqdm

class Engine:

    def __init__(self, device: str, epochs: int) -> None:
        self.device = device
        self.epochs = epochs

    def train_step(self,
                   model: Module,
                   loss_fn: Module,
                   optimizer: Optimizer,
                   dataloader: DataLoader):
        
        model.train()
        
        train_loss, train_acc = 0, 0
        
        for batch, (X, y) in enumerate(dataloader):

            X, y = X.to(self.device), y.to(self.device)

            y_pred = model(X)
            print(f"y_pred: {y_pred[0][0].shape}")

            # y_pred = y_pred[0][0][0]
            # print(f"y_pred: {y_pred}")

            # 2. Calculate  and accumulate loss
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            print(f"train_loss: {loss.item()}")

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            # Calculate and accumulate accuracy metric across all batches
            # y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            # train_acc += (y_pred_class == y).sum().item()/len(y_pred)

        train_loss = train_loss / len(dataloader)
        # train_acc = train_acc / len(dataloader)
        return train_loss, train_acc

    def test_step(self,
                  model: Module,
                  dataloader: DataLoader,
                  loss_fn: Module):
        
        test_loss, test_acc = 0, 0

        model.eval() 
        with torch.inference_mode():
            for batch, (X, y) in enumerate(dataloader):
                
                X, y = X.to(self.device), y.to(self.device)

                X = torch.permute(X, (1, 0, 2, 3, 4))

                for x in X:
            
                    # 1. Forward pass
                    test_pred_logits = model(x)
                    print(f"test_pred_logits: {test_pred_logits}")

                    # 2. Calculate and accumulate loss
                    loss = loss_fn(test_pred_logits, y)
                    # test_loss_img += loss.item()
                    # test_loss += loss.item()
                    # print(f"test_loss: {loss.item()}")
                    
                    # Calculate and accumulate accuracy
                    # test_pred_labels = test_pred_logits.argmax(dim=1)
                    # test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
                
        # test_loss = test_loss / len(dataloader)
        # test_acc = test_acc / len(dataloader)
        return test_loss, test_acc

    def train(self,
              model: Module,
              optimizer: Optimizer,
              loss_fn: Module,
              train_dataloader: DataLoader,
              test_dataloader: DataLoader):

        results = {"train_loss": [],
                   "train_acc": [],
                   "test_loss": [],
                   "test_acc": []}
        
        for epoch in tqdm(range(self.epochs)):

            train_loss, train_acc = self.train_step(model=model,
                                                    loss_fn=loss_fn,
                                                    optimizer=optimizer,
                                                    dataloader=train_dataloader)
            
            # test_loss, test_acc = self.test_step(model=model,
            #                                      dataloader=test_dataloader,
            #                                      loss_fn=loss_fn)
            
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                # f"test_loss: {test_loss:.4f} | "
                # f"test_acc: {test_acc:.4f}"
            )

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            # results["test_loss"].append(test_loss)
            # results["test_acc"].append(test_acc)

        return results