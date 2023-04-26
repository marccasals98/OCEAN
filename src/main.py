import torch
from torch.utils.data import DataLoader

from dataset import BlueFinLib
from ResNet import ResNet50, ResNet101, ResNet152
import torch.optim as optim
from utils import accuracy
import torch.nn.functional as F
import numpy as np

# fede


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train_single_epoch(model, train_loader, optimizer):
    model.train()
    accs, losses = [], []
    for x, y in train_loader:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        y_ = model(x)
        loss = F.cross_entropy(y_, y)
        loss.backward()
        optimizer.step()
        acc = accuracy(y, y_)
        losses.append(loss.item())
        accs.append(acc.item())
    return np.mean(accs), np.mean(losses)


def eval_single_epoch(model, val_loader):
    '''
    This function is made for both validation and test.
    '''
    accs, losses = [], []
    with torch.no_grad():
        model.eval()
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_ = model(x)
            loss = F.cross_entropy(y_, y)
            acc = accuracy(y, y_)
            losses.append(loss.item())
            accs.append(acc.item())
    return np.mean(accs), np.mean(losses)

def data_loaders():
    all_data = BlueFinLib()
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(all_data, [10000, 2500, 2500])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    return train_loader, val_loader, test_loader

def train_model(config):

    train_loader, val_loader, test_loader = data_loaders()

    my_model = ResNet50().to(device)
    optimizer = optim.Adam(my_model.parameters(), config["lr"])

    # TRAINING
    for epoch in range(config["epochs"]):
        loss, acc = train_single_epoch(my_model, train_loader, optimizer)
        print(f"Train Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
        loss, acc = eval_single_epoch(my_model, val_loader)
        print(f"Eval Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
    
    # TEST
    loss, acc = eval_single_epoch(my_model, test_loader)
    print(f"Test loss={loss:.2f} acc={acc:.2f}")

    return my_model

if __name__ == "__main__":

    config = {
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 10,
    }
    my_model = train_model(config)