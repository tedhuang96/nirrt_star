import time
import os
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader

from unet.models.unet_model import UNet
from unet.image_dataset import ImageDataset

########## LOAD CONFIGURATION ##########
batch_size = 32
img_height, img_width = 224, 224
path_thickness = 3
device = 'cuda'
lr = 1e-3
num_epochs = 100

########## LOAD DATA ##########
dset, dloader = {}, {}
for mode in ['train', 'val']:
    if mode == 'train':
        augment = True
        shuffle = True
    else:
        augment = False
        shuffle = False
    dset[mode] = ImageDataset(
        mode=mode,
        img_height=img_height,
        img_width=img_width,
        path_thickness=path_thickness,
        augment=augment,
    )
    dloader[mode] = DataLoader(
        dset[mode],
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
    )
print("Datasets are loaded.")

########## Initialize Model ##########
model = UNet().to(device)

########## Training ##########
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.functional.nll_loss
training_record = [["epoch", "train loss", "val loss", "train path IoU", "val path IoU", "epoch time"]]

weights = torch.Tensor(dset['train'].labelweights).to(device)

folder_path = "results/model_training/unet"
checkpoint_folderpath = folder_path + '/checkpoints'
log_folderpath = folder_path + '/logs'
os.makedirs(checkpoint_folderpath, exist_ok=True)
os.makedirs(log_folderpath, exist_ok=True)

best_val_acc = None

for epoch in range(1, num_epochs+1):
    epoch_start_time = time.time()
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    # train
    model.train()
    train_start_time = time.time()
    for batch_idx, batch in enumerate(dloader['train']):
        if (batch_idx+1) % 10 == 0:
            time_left = (time.time() - train_start_time) * (len(dloader['train']) / (batch_idx + 1) - 1) / 60
            print("Training {0}/{1}, remaining time: {2} min".format(batch_idx+1, len(dloader['train']), int(time_left)))
            print("Current training loss: {0:.4f}".format(np.mean(train_loss)))
        raw_img_input, img_input, img_label, token = batch
        img_input, img_label = img_input.to(device), img_label.to(device)

        optimizer.zero_grad()
        img_pred_logits = model(img_input) # (b,2,224,224)

        img_pred_logits = nn.functional.log_softmax(img_pred_logits, dim=1) # (b,2,224,224)
        img_pred_logits = img_pred_logits.permute(0,2,3,1).reshape(-1, 2) # (b*h*w, 2)
        img_label = img_label.reshape(-1).long()# (b*h*w,)
        loss = nn.functional.nll_loss(img_pred_logits, img_label, weights)
    
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        img_pred_flattened = img_pred_logits.detach().to('cpu').data.max(1)[1].float().reshape(batch_size, -1)
        img_label_flattened = img_label.detach().to('cpu').reshape(batch_size, -1)
        path_IoU = ((img_pred_flattened*img_label_flattened).sum(1)/(((img_pred_flattened+img_label_flattened)>0).float().sum(1)+1e-8)).tolist()
        train_acc += path_IoU
    
    # eval
    model.eval()
    val_start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dloader['val']):
            if (batch_idx+1) % 10 == 0:
                time_left = (time.time() - val_start_time) * (len(dloader['val']) / (batch_idx + 1) - 1) / 60
                print("Validating {0}/{1}, remaining time: {2} min".format(batch_idx+1, len(dloader['val']), int(time_left)))
                print("Curren validation loss: {0:.4f}".format(np.mean(val_loss)))
                # break
            raw_img_input, img_input, img_label, token = batch
            img_input, img_label = img_input.to(device), img_label.to(device)
            img_pred_logits = model(img_input)

            img_pred_logits = nn.functional.log_softmax(img_pred_logits, dim=1) # (b,2,224,224)
            img_pred_logits = img_pred_logits.permute(0,2,3,1).reshape(-1, 2) # (b*h*w, 2)
            img_label = img_label.reshape(-1).long()# (b*h*w,)
            loss = nn.functional.nll_loss(img_pred_logits, img_label, weights)
    

            val_loss.append(loss.item())
            img_pred_flattened = img_pred_logits.detach().to('cpu').data.max(1)[1].float().reshape(batch_size, -1)
            img_label_flattened = img_label.detach().to('cpu').reshape(batch_size, -1)
            path_IoU = ((img_pred_flattened*img_label_flattened).sum(1)/(((img_pred_flattened+img_label_flattened)>0).float().sum(1)+1e-8)).tolist()
            val_acc += path_IoU
    train_loss = np.mean(train_loss)
    val_loss = np.mean(val_loss)
    train_acc = np.mean(train_acc)
    val_acc = np.mean(val_acc)


    epoch_time = time.time()-epoch_start_time
    print("epoch {0}, train_loss: {1:.3f}, val_loss: {2:.3f}, train_path_IoU: {3:.3f}, val_path_IoU: {4:.3f}, time: {5} sec"\
        .format(epoch, train_loss, val_loss, train_acc, val_acc, int(epoch_time)))
    if best_val_acc is None or val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), checkpoint_folderpath+'/best_unet.pt')
    training_record_epoch = [
        "{0}".format(epoch), \
        "{0:.3f}".format(train_loss),
        "{0:.3f}".format(val_loss), \
        "{0:.3f}".format(train_acc), \
        "{0:.3f}".format(val_acc), \
        "{0}".format(int(epoch_time))]
    training_record.append(training_record_epoch)
    np.savetxt(log_folderpath+"/train_record.csv", np.stack(training_record, axis=0), delimiter=",", fmt='%s')