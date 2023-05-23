import torch
from torch.utils.data import DataLoader
from dataset import BlueFinLib
from ResNet import ResNet50, ResNet101, ResNet152
import torch.optim as optim
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def accuracy(labels: torch.Tensor, outputs: torch.Tensor) -> int:
    '''
    This function returns the number of coincidences that happen in two arrays of the same length.

    Arguments:
    ----------
    labels : torch.Tensor [batch, num_classes]
        The ground truth of the classes.
    outputs : torch.Tensor [batch, num_classes]
        The model prediction of the most likely class.
    
    Returns:
    --------
    acum : int
        The number of coincidences.
    '''
    preds = outputs.argmax(-1, keepdim=True)
    labels = labels.argmax(-1, keepdim=True) # bc we have done one_hot encoding.
    # label shape [batch, 1], outputs shape [batch, 1]
    acum = preds.eq(labels.view_as(preds)).sum().item() # sums the times both arrays coincide.
    return acum

def train_single_epoch(model, train_loader, optimizer):
    model.train()
    accs, losses = [], []
    for x, y in tqdm(train_loader, unit="batch", total=len(train_loader)):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        y_ = model(x)
        #print('output: ', y_)
        #print('labels: ', y)
        loss = F.cross_entropy(y_, y)
        loss.backward()
        optimizer.step()
        acc = accuracy(y, y_)
        losses.append(loss.item())
        accs.append(acc) # accs.append(acc.item())
    return np.mean(losses), np.sum(accs)/len(train_loader.dataset)


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
            accs.append(acc) # accs.append(acc.item())
            #pred = y_.detach().numpy()
            #cm = confusion_matrix(y.argmax(-1), pred.argmax(-1))
            #print('Confussion matrix eval:\n', cm) # maybe not necessary to be print every time.
    return  np.mean(losses), np.sum(accs)/len(val_loader.dataset)

def data_loaders(config):
    data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    total_data = BlueFinLib(pickle_path = "/home/usuaris/veussd/DATABASES/Ocean/df_23_05_21_12_08_09_23hqmc53_zany-totem-48.pkl", 
                            img_dir = "/home/usuaris/veussd/DATABASES/Ocean/Spectrograms_AcousticTrends/23_05_21_12_08_09_23hqmc53_zany-totem-48", 
                            config = config,
                            transform=data_transforms)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(total_data,
                                                                              [config['num_samples_train'],
                                                                                config['num_samples_val'],
                                                                                config['num_samples_test']])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    return train_loader, val_loader, test_loader

def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", "pred", "target"]+[f"score_{i}" for i in range(10)])
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
    wandb.log({"predictions_table":table}, commit=False)

def wandb_init(config):
    wandb.init(project="acoustic_trends", config=config)
    wandb.run.name = f"{config['architecture']}_lr={config['lr']}_bs={config['batch_size']}_epochs={config['epochs']}"
    wandb.run.save()

def train_model(config):

    train_loader, val_loader, test_loader = data_loaders(config)

    my_model = ResNet50(num_classes = 5, channels=1).to(device)
    optimizer = optim.Adam(my_model.parameters(), config["lr"])
    wandb_init(config)

    # TRAINING
    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_single_epoch(my_model, train_loader, optimizer)
        print(f"Train Epoch {epoch} loss={train_loss:.2f} acc={train_acc:.2f}")
        val_loss, val_acc = eval_single_epoch(my_model, val_loader)
        print(f"Eval Epoch {epoch} loss={val_loss:.2f} acc={val_acc:.2f}")
        train_metrics = {"train/train_loss":train_loss,
                        "train/train_acc":train_acc,
                        "val/val_loss":val_loss,
                        "val/val_acc":val_acc}
        wandb.log(train_metrics, step=epoch+1)

    # TEST
    loss, acc = eval_single_epoch(my_model, test_loader)
    print(f"Test loss={loss:.2f} acc={acc:.2f}")
    wandb.log({"test/test_loss":loss,
                "test/test_acc":acc})

    wandb.finish()
    return my_model


if __name__ == "__main__":
    # TODO: calculate properly "random_crop_frames". Use the fede function
    
    config = {
        "lr": 1e-3,
        "batch_size": 3, # This number must be bigger than one (nn.BatchNorm)
        "epochs": 10,
        "architecture": "ResNet50",
        "num_samples_train": 57532,
        "num_samples_val": 19178,
        "num_samples_test": 19177,
        "random_crop_frames": 4,
    }
    my_model = train_model(config)
