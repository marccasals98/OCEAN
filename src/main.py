import torch
from torch.utils.data import DataLoader
from dataset import BlueFinLib
from ResNet import ResNet50, ResNet101, ResNet152
# import DMHA
from DMHA import SpeakerClassifier
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

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

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
        # print(device)
        x, y = x.to(device), y.to(device)
        # print(f'input: {x.size()}')
        # print('labels: ', y.size())
        y_ = model(x, y)[0] #                   POL: AFEGIT [0]
        # print('output: ', y_.size())
        loss = F.cross_entropy(y_, y)
        # print(f'loss:{loss}')
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
            y_ = model(x,y)[0] #                                  POL: HE POSAT ",y"
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
    total_data = BlueFinLib(pickle_path = config['df_path'], 
                            img_dir = config['img_dir'], 
                            config = config,
                            transform=data_transforms)

    #nou. TOT AQUEST BLOC ES LO NOU, BORRAR I DESCOMENTAR SOTA
    num_samples = len(total_data)
    train_size = int(num_samples * config["num_samples_train"])
    val_size = int(num_samples * config["num_samples_val"])
    test_size = num_samples - train_size - val_size
    train_data, val_data, test_data = torch.utils.data.random_split(total_data, [train_size, val_size, test_size])
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"])
    test_loader = DataLoader(test_data, batch_size=config["batch_size"])

    '''train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(total_data,
                                                                              [config['num_samples_train'],
                                                                                config['num_samples_val'],
                                                                                config['num_samples_test']])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])'''

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

def select_model(config):
    if config['architecture'] == "ResNet50":
        model = ResNet50(num_classes = len(config['species']), channels=1).to(device)
    elif config['architecture'] == 'LeNet5':
        model = LeNet5(n_classes= len(config['species'])).to(device)
    elif config['architecture'] == 'SpeakerClassifier':
        model = SpeakerClassifier(config, device).to(device)
    else:
        raise ValueError('The model name is not on the list')
    return model

def train_model(config):

    model_name = f"{config['architecture']}_lr{config['lr']}_bs{config['batch_size']}_epochs{config['epochs']}"
    print("=" * 60)
    print('Running model:', model_name)
    print("=" * 60)

    train_loader, val_loader, test_loader = data_loaders(config)

    #nou. 
    #best_metric = float('-inf')
    #best_params = None
    #nou. Lists to store results across iterations
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    test_losses = []
    test_accuracies = []
    #nounou. 
    best_val_acc = float('-inf')
    best_model_state = None

    #nou. Iterations
    for iteration in range(config["num_iterations"]):
        print(f"Iteration: {iteration+1}")

        my_model = select_model(config)
        optimizer = optim.Adam(my_model.parameters(), config["lr"])
        wandb_init(config)
        # best_metric = float('-inf') nou. descomentar ...
        #best_params = None nou. descomentar per ser com abans

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
            wandb.log(train_metrics, step=epoch+1) #nounounou. comentat
            '''if val_acc > best_metric:
                best_metric = val_acc
                best_params = my_model.state_dict()
                #torch.save(best_params, config["save_dir"] + f"{config['architecture']}_lr{config['lr']}_bs{config['batch_size']}_epochs{config['epochs']}.pt")
                torch.save(best_params, "/home/usuaris/veu/pol.cavero/OCEAN/save_best_modelPol/" + model_name + ".pt")'''
            
            # nounou. save millor model de totes les iteracions
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = my_model.state_dict()
                torch.save(best_model_state, "/home/usuaris/veu/pol.cavero/OCEAN/save_best_modelPol/" + model_name + ".pt")
                # wandb.log(train_metrics, step=epoch+1)

        # TEST
        loss, acc = eval_single_epoch(my_model, test_loader)
        print(f"Test loss={loss:.2f} acc={acc:.2f}")

        wandb.log({"test/test_loss":loss,
                    "test/test_acc":acc})

        # nou. Store results for comparison
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        test_losses.append(loss)
        test_accuracies.append(acc)
    
    # nou. Compare the results across iterations
    print("Train Losses:", train_losses)
    print("Train Accuracies:", train_accuracies)
    print("Average Train Loss:", sum(train_losses)/len(train_losses))
    print("Average Train Accuracy:", sum(train_accuracies)/len(train_accuracies))
    print("Validation Losses:", val_losses)
    print("Validation Accuracies:", val_accuracies)
    print("Average Validation Loss:", sum(val_losses)/len(val_losses))
    print("Average Validation Accuracy:", sum(val_accuracies)/len(val_accuracies))
    print("Test Losses:", test_losses)
    print("Test Accuracies:", test_accuracies)
    print("Average Test Loss:", sum(test_losses)/len(test_losses))
    print("Average Test Accuracy:", sum(test_accuracies)/len(test_accuracies))
    print(f"Best model: Val acc:{best_val_acc}")

    wandb.finish()
    return my_model


if __name__ == "__main__":
    # TODO: check if random_crop_frames are calculated properly!
    # TODO: implement the save model.
    # TODO: wandb.run.save without any arguments is deprecated. 

    '''config = {
        "architecture": "ResNet50",
        "lr": 1e-3,
        "batch_size": 64, # This number must be bigger than one (nn.BatchNorm)
        "epochs": 1,
        "num_samples_train": 0.8,
        "num_samples_val": 0.1,
        "num_samples_test": 0.1,
        "species": ['Fin', 'Blue'],
        "random_crop_secs": 5, 
        "df_dir": "/home/usuaris/veussd/DATABASES/Ocean/dataframes",
        "df_path": "",
        "img_dir" : "/home/usuaris/veussd/DATABASES/Ocean/Spectrograms_AcousticTrends/23_06_02_09_07_26_aty1jmit_wise-meadow-57",
        "save_dir": "/home/usuaris/veussd/DATABASES/Ocean/checkpoints/"
    }'''

    # config for DMHA model
    config = {
        "architecture": "SpeakerClassifier", #canviat de ResNet50
        "lr": 1e-3,
        "batch_size": 128, # Marc tenia 60
        "epochs": 2,
        "num_samples_train": 0.8,
        "num_samples_val": 0.1,
        "num_samples_test": 0.1,
        "species": ['Fin', 'Blue'],
        "random_crop_secs": 5,
        "df_dir": "/home/usuaris/veussd/DATABASES/Ocean/dataframes",
        "df_path": "",
        "img_dir" : "/home/usuaris/veussd/DATABASES/Ocean/Spectrograms_AcousticTrends/23_06_02_09_07_26_aty1jmit_wise-meadow-57",
        "save_dir": "/home/usuaris/veussd/DATABASES/Ocean/checkpoints/",
        
        # afegim necessaris per DMHA
        "train_labels_path": '/home/usuaris/veu/pol.cavero/labels_DMHA_Voxceleb/train_labels.ndx',
        "train_data_dir": '/home/usuaris/veu/pol.cavero/datasets/voxceleb_2/dev/23_04_15_16_23_42_svagf2q9_noble-water-4/',
        "valid_clients_path": '/home/usuaris/veu/pol.cavero/labels_DMHA_Voxceleb/valid_clients.ndx',
        "valid_impostors_path": '/home/usuaris/veu/pol.cavero/labels_DMHA_Voxceleb/valid_impostors.ndx',
        "valid_data_dir": '/home/usuaris/veu/pol.cavero/datasets/voxceleb_2/dev/23_04_15_16_23_42_svagf2q9_noble-water-4/',
        "model_output_folder": './models/',
        "embedding_size": 400,
        "front_end": 'VGGNL',
        "pooling_method": 'DoubleMHA',
        "pooling_heads_number": 5, # Fede, abans era 32
        "pooling_mask_prob": 0.000001,

        # inventat numero de mels
        "n_mels": 40, # Fede
        "pooling_output_size": 400, 
        "bottleneck_drop_out": 0.1, 
        # DMHA params
        "vgg_n_blocks": 3, # Fede (abans 1)
        "vgg_channels": [16, 32, 64], # Fede (abans [1])
        "number_speakers": 2, # Inventat, pero dos classes == dos speakers
        "scaling_factor": 30, # M'ho he inventat perque no doni error
        "margin_factor": 0.4,
        
        # crossvalidation
        "num_iterations": 4, # posat per mi

    }

    df_creator = DataframeCreator(config['img_dir'], config['df_dir'])
    config["df_path"] = df_creator.get_df_path()

    my_model = train_model(config)
