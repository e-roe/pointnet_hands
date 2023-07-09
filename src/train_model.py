
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import mediapipe as mp
from point_net import PointNet
import matplotlib.pyplot as plt
from torch import optim
from utils import load_config, char2int

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PointsDataSet(Dataset):
    def __init__(self, path, items):
        self.path = path
        self.items = items

    def __getitem__(self, item):
        points = np.load(os.path.join(self.path, self.items[item][0], self.items[item][1]))
        points = np.array(points)

        return points, char2int[self.items[item][0]]

    def __len__(self):
        return len(self.items)

    def collate_fn(self, batch):
        ims, classes = list(zip(*batch))

        ims = torch.cat([torch.Tensor(im) for im in ims]).float().to(device)
        ims = torch.reshape(ims, (10, 21, 3))
        ce_masks = torch.cat([torch.Tensor([cla]) for cla in classes]).long().to(device)

        return ims, ce_masks


def split_files(path, divs):
    if divs[0] + divs[1] + divs[2] != 1.0:
        print(f'Wrong divisions: Train={divs[0]} Validation={divs[1]} Test={divs[2]} Total={divs[0] + divs[1] + divs[2]}')
        sys.exit()

    items = []
    for root, dirs, files in os.walk(path, topdown=True):
        for file in files:
            items.append((root.split(os.sep)[-1], file))

    np.random.shuffle(items)
    size = len(items)
    train = int(size * divs[0])
    val = int(size * divs[1])
    test = int(size * divs[2])
    total = train + test + val
    train = train + (size - total)

    #train - validation - test
    return items[:train], items[train:train+val], items[train+val:]


def train_batch(model, data, optmz, loss):
    model.train()
    points, classes = data
    _masks = model(points)
    optmz.zero_grad()
    loss, acc = loss(_masks, classes)
    loss.backward()
    optmz.step()

    return loss.item(), acc.item()


@torch.no_grad()
def validate_batch(model, data, criterion):
    model.eval()
    ims, masks = data
    _masks = model(ims)
    loss, acc = criterion(_masks, masks)

    return loss.item(), acc.item()


def pointnet_loss(preds, targets):
    ce = nn.CrossEntropyLoss()
    ce_loss = ce(preds, targets)
    acc = (torch.max(preds, 1)[1] == targets).float().mean()
    return ce_loss, acc


def main():
    config = load_config('config.yaml')
    count = 0
    model_path = config['model']['model_path']
    os.makedirs(model_path, exist_ok=True)
    results_path = config['paths']['results_path']
    os.makedirs(results_path, exist_ok=True)
    path = config['dataset']['npy_dataset']
    model_name = config['model']['name']
    with open(os.path.join(results_path, 'status.csv'), 'w') as f:
        f.write(f'Loss train; Loss val; Acc train; Acc val\n')

    train_p = config['dataset']['train_percent']
    validation_p = config['dataset']['validation_percent']
    test_p = config['dataset']['test_percent']
    train_files, val_files, _ = split_files(path, (train_p, validation_p, test_p))

    n_epochs = config['trainer']['num_epochs']
    batch_size = config['trainer']['batch_size']

    train_ds = PointsDataSet(path, train_files)
    val_ds = PointsDataSet(path, val_files)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_ds.collate_fn, drop_last = True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=val_ds.collate_fn, drop_last = True)

    model = PointNet(len(char2int)).to(device)
    l_rate = config['optimizer']['learning_rate']
    loss_function = pointnet_loss
    optimizer_class = getattr(optim, config['optimizer']['name'])
    optimizer = optimizer_class(model.parameters(), lr=l_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=0.5, patience=0,
                                                     threshold=0.001, verbose=True,
                                                     min_lr=1e-5, threshold_mode='abs')
    loss_train_all = []
    acc_train_all = []
    loss_val_all = []
    acc_val_all = []
    for epoch in range(n_epochs):
        print(f'Current epoch:{epoch}')
        loss_ep = []
        acc_ep = []
        for points, classes in enumerate(train_dl):
            loss, acc = train_batch(model, classes, optimizer, loss_function)
            loss_ep.append(loss)
            acc_ep.append(acc)
        loss_train_all.append(np.mean(loss_ep))
        acc_train_all.append(np.mean(acc_ep))
        loss_ep = []
        acc_ep = []
        for points, classes in enumerate(val_dl):
            loss, acc = validate_batch(model, classes, loss_function)
            acc_ep.append(acc)
            loss_ep.append(loss)
        loss_val_all.append(np.mean(loss_ep))
        acc_val_all.append(np.mean(acc_ep))
        val_loss = np.mean(loss_ep)
        scheduler.step(val_loss)

        print(f'Loss train {loss_train_all[-1]} Loss val {loss_val_all[-1]} Acc train {acc_train_all[-1]} Acc val {acc_val_all[-1]}')
        with open(os.path.join(results_path, 'status.csv'), 'a') as f:
            f.write(f'{loss_train_all[-1]}; {loss_val_all[-1]}; {acc_train_all[-1]}; {acc_val_all[-1]}\n')

        count += 1
        if count % 10 == 0:
            torch.save(model, os.path.join(model_path, model_name))

    torch.save(model, os.path.join(model_path, model_name))

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    axs[0].plot(loss_train_all, color='green',  label=['Loss Train'])
    axs[0].plot(loss_val_all, color='red',  label=['Loss val'])
    axs[0].set_title("Loss")
    axs[0].legend()
    plt.legend()

    axs[1].plot(acc_train_all, color='green',  label=['Acc Train'])
    axs[1].set_title("Accuracy")
    axs[1].plot(acc_val_all, color='red',  label=['Acc Val'])
    axs[1].legend()
    plt.savefig(os.path.join(results_path, f'metrics_point_net_NPY_SCLNoFlip_{n_epochs}epochs_{l_rate}lr4.png'))

    plt.show()


if __name__ == '__main__':
    main()