import torch
from torch.utils.data import DataLoader

from ResNet import ResNet50, ResNet101, ResNet152
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train_single_epoch(model, train_loader, optimizer):
    return 1


def eval_single_epoch(model, val_loader):
    return 1


def train_model(config):

    my_model = ResNet50().to(device)

    optimizer = optim.Adam(my_model.parameters(), config["lr"])
    for epoch in range(config["epochs"]):
        loss, acc = train_single_epoch(my_model, train_loader, optimizer)
        print(f"Train Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
        loss, acc = eval_single_epoch(my_model, val_loader)
        print(f"Eval Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
    
    loss, acc = eval_single_epoch(my_model, test_loader)
    print(f"Test loss={loss:.2f} acc={acc:.2f}")

    return my_model



if __name__ == "__main__":

    config = {
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 5,
        "h1": 32,
        "h2": 64,
        "h3": 128,
        "h4": 128,
    }
    my_model = train_model(config)