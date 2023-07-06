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
from DataframeCreator import DataframeCreator
import os
from LeNet import LeNet5
from metrics import accuracy, Metrics


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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


def eval_single_epoch(model, val_loader, test=False):
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
            
            # Confussion Matrix:
            if test == True:
                pred = y_.cpu().detach().numpy()
                cm = confusion_matrix(y.cpu().argmax(-1), pred.argmax(-1))
                print('Confussion matrix eval:\n', cm) # maybe not necessary to be print every time.
            
            # TODO: FINISH THIS!!!!!
            """
            # Other metrics:
            metric = Metrics(labels=y.cpu().argmax(-1), outputs=pred.argmax(-1), device=device)
            precision = metric.precision()
            """
            precision = 1
    return  np.mean(losses), np.sum(accs)/len(val_loader.dataset), precision

def data_loaders(config):
    data_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(0.5, 0.5),
                                        transforms.RandomErasing(p=config["random_erasing"]),])
    total_data = BlueFinLib(pickle_path = config['df_path'], 
                            img_dir = config['img_dir'], 
                            config = config,
                            transform=data_transforms)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(total_data,
                                                                              [config['num_samples_train'],
                                                                                config['num_samples_val'],
                                                                                config['num_samples_test']])
    # TODO: Implement data augmentation.
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
    wandb.run.name = f"{config['architecture']}_lr={config['lr']}_bs={config['batch_size']}_epochs={config['epochs']}_random_crop_secs{config['random_crop_secs']}_random_erasing{config['random_erasing']}"
    wandb.run.save()

def select_model(config):
    if config['architecture'] == "ResNet50":
        model = ResNet50(num_classes = len(config['species']), channels=1).to(device)
    elif config['architecture'] == 'LeNet5':
        model = LeNet5(n_classes= len(config['species'])).to(device)
    else:
        raise ValueError('The model name is not on the list')
    return model

def train_model(config):

    model_name = f"{config['architecture']}_lr{config['lr']}_bs{config['batch_size']}_epochs{config['epochs']}_random_crop_secs{config['random_crop_secs']}_random_erasing{config['random_erasing']}"
    print("=" * 60)
    print('Running model:', model_name)
    print("=" * 60)

    train_loader, val_loader, test_loader = data_loaders(config)
    my_model = select_model(config)
    optimizer = optim.Adam(my_model.parameters(), config["lr"])
    wandb_init(config)
    best_metric = float('-inf')
    best_params = None
    best_epoch = 0

    # TRAINING
    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_single_epoch(my_model, train_loader, optimizer)
        print(f"Train Epoch {epoch+1} loss={train_loss:.2f} acc={train_acc:.2f}")
        val_loss, val_acc, pre = eval_single_epoch(my_model, val_loader)
        print(f"Eval Epoch {epoch+1} loss={val_loss:.2f} acc={val_acc:.2f}")
        print(f"The precision is {pre} ")
        train_metrics = {"train/train_loss":train_loss,
                        "train/train_acc":train_acc,
                        "val/val_loss":val_loss,
                        "val/val_acc":val_acc}
        wandb.log(train_metrics, step=epoch+1)
        if val_acc > best_metric:
            # TODO: Print the best validation score.∫
            best_epoch = epoch
            best_metric = val_acc
            best_params = my_model.state_dict()
            # torch.save(best_params, config["save_dir"] + f"{config['architecture']}_lr{config['lr']}_bs{config['batch_size']}_epochs{config['epochs']}.pt")
            torch.save(best_params, "/home/usuaris/veu/marc.casals/ocean/" + model_name + ".pt")
        

    # TEST
    my_model.load_state_dict(best_params) # load the best params of the validation.
    loss, acc, pre = eval_single_epoch(my_model, test_loader, test=True)
    print(f"Test loss={loss:.2f} acc={acc:.2f}")
    wandb.log({"test/test_loss":loss,
                "test/test_acc":acc})
    print(f"The best epoch is epoch {best_epoch+1}")

    wandb.finish()
    return my_model


if __name__ == "__main__":
    # TODO: wandb.run.save without any arguments is deprecated. 
    config = {
        "architecture": "ResNet50",
        "lr": 1e-3,
        "batch_size": 64, # This number must be bigger than one (nn.BatchNorm).
        "epochs": 1,
        "num_samples_train": 0.6,
        "num_samples_val": 0.2,
        "num_samples_test": 0.2,
        "species": ['Fin', 'Blue'],
        "random_crop_secs": 5, # number of seconds that has the spectrogram.
        "random_erasing": 0, # probability that the random erasing operation will be performed.
        "df_dir": "/home/usuaris/veussd/DATABASES/Ocean", # where the pickle dataframe is stored.
        "df_path": "",
        "img_dir" : "/home/usuaris/veussd/DATABASES/Ocean/toyDataset", # directory of the spectrograms.
        "save_dir": "/home/usuaris/veussd/DATABASES/Ocean/checkpoints/" # where we save the model checkpoints.
    }

    df_creator = DataframeCreator(config['img_dir'], config['df_dir'])
    config["df_path"] = df_creator.get_df_path()

    my_model = train_model(config)
