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
from metrics import accuracy, Metrics, plot_confusion_matrix


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train_single_epoch(model, train_loader, optimizer):
    model.train()
    accs, losses = [], []
    for x, y in tqdm(train_loader, unit="batch", total=len(train_loader)):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        y_ = model(x)
        loss = F.cross_entropy(y_, y)
        loss.backward()
        optimizer.step()
        acc = accuracy(y, y_)
        losses.append(loss.item())
        accs.append(acc) # accs.append(acc.item())
    return np.mean(losses), np.sum(accs)/len(train_loader.dataset)


def eval_single_epoch(model, val_loader, config, test=False):
    '''
    This function is made for both validation and test.
    '''
    accs, losses, precisions, recalls, f1s = [], [], [], [], []
    cm = np.empty([len(config['species']), len(config['species'])])
    with torch.no_grad():
        model.eval()
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_ = model(x)
            loss = F.cross_entropy(y_, y)
            acc = accuracy(y, y_)
            losses.append(loss.item())
            accs.append(acc) # accs.append(acc.item())
            
            # We don't want to print all these metrics in evaluation, just in test.
            if test == True:
                # Confussion Matrix:
                pred = y_.cpu().detach().numpy()
                cm = cm + confusion_matrix(y.cpu().argmax(-1), pred.argmax(-1))
            
                # Other metrics:
                metric = Metrics(labels=y, outputs=y_, config=config, device=device)
                metric.compute_metrics()
                precisions.append(metric.precision)
                recalls.append(metric.recall)
                f1s.append(metric.f1)
    if test == True:
        return  np.mean(losses), np.sum(accs)/len(val_loader.dataset), torch.mean(torch.stack(precisions)), torch.mean(torch.stack(recalls)), torch.mean(torch.stack(f1s)), cm
    else:
        return  np.mean(losses), np.sum(accs)/len(val_loader.dataset)
            

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
    model_name = f"{config['architecture']}_lr={config['lr']}_bs={config['batch_size']}_epochs={config['epochs']}_random_crop_secs{config['random_crop_secs']}_random_erasing{config['random_erasing']}"
    wandb.run.name = model_name
    wandb.run.save(f"{model_name}.h5")

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
        val_loss, val_acc = eval_single_epoch(my_model, val_loader, config)
        print(f"Eval Epoch {epoch+1} loss={val_loss:.2f} acc={val_acc:.2f}")
        train_metrics = {"train/train_loss":train_loss,
                        "train/train_acc":train_acc,
                        "val/val_loss":val_loss,
                        "val/val_acc":val_acc}
        wandb.log(train_metrics, step=epoch+1)
        if val_acc > best_metric:
            best_epoch = epoch
            best_metric = val_acc
            best_params = my_model.state_dict()
            # For each best validation, we overwrite the model parameters.
            # The model could stop training and we'd still have the best params safe.
            torch.save(best_params, "/home/usuaris/veu/marc.casals/ocean/" + model_name + ".pt") 
        

    # TEST
    my_model.load_state_dict(best_params) # load the best params of the validation.
    loss, acc, pre, recall, f1, cm = eval_single_epoch(my_model, test_loader, config, test=True)
    
    print('Confussion matrix test:\n', cm)
    # seaborn confussion matrix
    save_cm = '/home/usuaris/veu/marc.casals/OCEAN/plots/confussion_matrix.png'
    plot_confusion_matrix(cm, len(config['species']), save_cm)
    # loading metrics in wandb
    wandb.log({"test/test_loss":loss, 
                "test/test_acc":acc,
                "test/test_precision":pre,
                "test/test_recall":recall,
                "test/test_f1":f1})
    print(f"The best epoch is epoch {best_epoch+1}")

    wandb.finish()
    return my_model


if __name__ == "__main__":
    # TODO: wandb.run.save without any arguments is deprecated. 
    config = {
        "architecture": "LeNet5",
        "lr": 1e-3,
        "batch_size": 64, # This number must be bigger than one (nn.BatchNorm).
        "epochs": 1,
        "num_samples_train": 0.6,
        "num_samples_val": 0.2,
        "num_samples_test": 0.2,
        "species": ['Fin', 'Blue'],
        "random_crop_secs": 5, # number of seconds that has the spectrogram.
        "random_erasing": 0, # probability that the random erasing operation will be performed.
        "df_dir": "/home/usuaris/veussd/DATABASES/Ocean/dataframes", # where the pickle dataframe is stored.
        "df_path": "",
        "img_dir" : "/home/usuaris/veussd/DATABASES/Ocean/Spectrograms_AcousticTrends/23_06_02_09_07_26_aty1jmit_wise-meadow-57", # directory of the spectrograms.
        "save_dir": "/home/usuaris/veussd/DATABASES/Ocean/checkpoints/", # where we save the model checkpoints.
        "train_specs": "/home/usuaris/veussd/DATABASES/Ocean/SPECTROGRAMS_MARC/ORIGINAL/TRAIN",
        "val_specs": "/home/usuaris/veussd/DATABASES/Ocean/SPECTROGRAMS_MARC/ORIGINAL/VALID",
        "test_specs": "/home/usuaris/veussd/DATABASES/Ocean/SPECTROGRAMS_MARC/ORIGINAL/TEST"
    }

    df_creator = DataframeCreator(config['img_dir'], config['df_dir'])
    config["df_path"] = df_creator.get_df_path()
    print(config["df_path"])

    df_creator_train = DataframeCreator(config['train_specs'], config['df_dir'])
    config["df_path_train"] = df_creator_train.get_df_path()

    df_creator_val = DataframeCreator(config['val_specs'], config['df_dir'])
    config["df_path_val"] = df_creator_val.get_df_path()

    df_creator_test = DataframeCreator(config['test_specs'], config['df_dir'])
    config["df_path_test"] = df_creator_test.get_df_path()

    #my_model = train_model(config)
