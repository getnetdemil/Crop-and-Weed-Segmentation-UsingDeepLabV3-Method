
from typing import Callable, List
import torch
import torch.utils.data as data


class BaselineTrainer:
    def __init__(self, model: torch.nn.Module, loss: Callable, optimizer: torch.optim.Optimizer, device="cpu"):

        self.model = model.to(device)
        # self.model = model
        
        self.loss = loss
        self.optimizer = optimizer
        self.device = device

#Loading.
    def fit(self, train_data_loader: data.DataLoader, epochs: int):
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = len(train_data_loader)

            print(f"Epoch {epoch + 1}/{epochs}")

            for batch_idx, (ref_img, dist_img) in enumerate(train_data_loader):
                ref_img, dist_img = ref_img.to(self.device), dist_img.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(dist_img)
                loss = self.loss(output, ref_img)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                print(f"\rBatch {batch_idx + 1}/{num_batches} - Loss: {loss.item():.4f}", end='')

            avg_loss = total_loss / num_batches
            print(f"\nEpoch {epoch + 1} Average Loss: {avg_loss:.4f}")

        print("Training finished.")
