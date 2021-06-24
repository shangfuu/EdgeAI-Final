import enum
from facenet_pytorch import MTCNN, InceptionResnetV1

import cv2
import numpy as np
import os
import time
import shutil
from tqdm import tqdm


import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torch import optim, nn
from torch.nn import functional as F

from torch.optim.lr_scheduler import MultiStepLR

import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as torchData

import config as cfg
import train_utils as train_func
from FRNet import FRNet

# Device Check
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


def train(save_path="models/model.pth", tune=False):
    # Data Transforms
    tfms = transforms.Compose([transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)), transforms.ToTensor()])

    if tune:
        dataset = datasets.ImageFolder(cfg.DATA_LFW_CROP, transform=tfms)
    else:
        dataset = datasets.ImageFolder(cfg.DATA_WEBFACE, transform=tfms)
    
    # Train and Valiate split
    train_set_size = int(len(dataset) * 0.8)
    valid_set_size = len(dataset) - train_set_size
    train_set, valid_set = torchData.random_split(dataset, [train_set_size, valid_set_size])

    if tune:
        # Class weight for imbalance data
        class_weights = []
        # DATA_LFW_CROP
        for root, subdir, files in os.walk(cfg.DATA_WEBFACE):
            if len(files) > 0:
                class_weights.append(1/len(files))
            else:
                class_weights.append(0)

        
        print("Train")
        train_weights = [0] * len(train_set)
        for idx, (_, label) in enumerate(train_set):
            train_weights[idx] = class_weights[label]
        # # train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

        print("Valid")
        valid_weights = [0] * len(valid_set)
        for idx, (_, label) in enumerate(valid_set):
            valid_weights[idx] = class_weights[label]
        # # valid_sampler = WeightedRandomSampler(valid_weights, num_samples=len(valid_weights), replacement=True)

        train_loader = DataLoader(
            train_set,
            num_workers=cfg.NUM_WORKERS,
            batch_size=cfg.BATCH_SIZE,
            # sampler=SubsetRandomSampler(train_inds)
            sampler=WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
        )
        val_loader = DataLoader(
            valid_set,
            num_workers=cfg.NUM_WORKERS,
            batch_size=cfg.BATCH_SIZE,
            # sampler=SubsetRandomSampler(val_inds)
            sampler=WeightedRandomSampler(valid_weights, num_samples=len(valid_weights), replacement=True)
        )
    else:

        train_loader = DataLoader(
            train_set,
            num_workers=cfg.NUM_WORKERS,
            batch_size=cfg.BATCH_SIZE,
            # sampler=SubsetRandomSampler(train_inds)
            # sampler=WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
        )
        val_loader = DataLoader(
            valid_set,
            num_workers=cfg.NUM_WORKERS,
            batch_size=cfg.BATCH_SIZE,
            # sampler=SubsetRandomSampler(val_inds)
            # sampler=WeightedRandomSampler(valid_weights, num_samples=len(valid_weights), replacement=True)
        )

    if tune:
        model = InceptionResnetV1(
            classify=True,
            pretrained='vggface2',
            num_classes=len(dataset.class_to_idx)
        ).to(device)

    else:
        model = FRNet(classify=True, pretrained=True, num_classes=len(dataset.class_to_idx)).to(device=device)

    loss_fn = torch.nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    scheduler = MultiStepLR(optimizer, [5, 10])

    metrics = {
        'fps': train_func.BatchTimer(),
        'acc': train_func.accuracy
    }
    
    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10

    
    print('-' * 10)
    
    # model.eval()
    # train_func.pass_epoch(
    #     model, loss_fn, val_loader, batch_metrics=metrics, show_running=True, device=device, writer=writer
    # )

    best_acc = 0.0

    for epoch in range(cfg.EPOCHS):
        print('\nEpoch {}/{}'.format(epoch + 1, cfg.EPOCHS))
        print('-' * 10)

        model.train()
        loss, met = train_func.pass_epoch(
            model, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

        model.eval()
        loss, met = train_func.pass_epoch(
            model, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

        current_acc = met['acc'].item()
        if current_acc > best_acc:
            print("Best Accuracy : {:4f}".format(current_acc))
            print("Save model...")
            best_acc = current_acc

            torch.save(model.state_dict(), save_path)
    
    writer.close()



def Data_check(data_folder=cfg.DATA_LFW ,makedir = False):

    # How many class
    print("Classes: ", len(os.listdir(data_folder)))

    # Total images
    total = sum([len(files) for r, d, files in os.walk(data_folder)])
    print("Total images: ", total)

    # class more than 5 images
    imglist=[]
    NEW_ROOT = "NEW_DATA/"
    for r, d, files in os.walk(data_folder):   
        if len(files)>=5: 
            imglist.append(r)

            if (makedir):
                base_name = os.path.basename(r)
                new_path = os.path.join(NEW_ROOT, base_name)
                shutil.copytree(r, new_path)
        
    print(len(imglist))
    # print("Folders: \n", imglist)



if __name__ == '__main__':

    # Data_check(data_folder=cfg.DATA_WEBFACE)
    # Data_check(data_folder=cfg.DATA_LFW_CROP)
    
    train(save_path="models/lfw_tune.pth", tune=True)
    # test()

    pass