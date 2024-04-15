import time
from os.path import exists

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from unet.models.unet_model import UNet
from unet.image_dataset import ImageDataset

########## LOAD CONFIGURATION ##########
batch_size = 16 # we want it to be uniform across all model evaluation. original: 32
img_height, img_width = 224, 224
path_thickness = 3
device = 'cuda'
lr = 1e-3
num_epochs = 100

########## LOAD DATA ##########
dset, dloader = {}, {}
dset['train'] = ImageDataset(
    mode='train',
    img_height=img_height,
    img_width=img_width,
    path_thickness=path_thickness,
    augment=True,
)
for mode in ['test']:
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
model_filename = 'results/model_training/unet/checkpoints/best_unet.pt'
if exists(model_filename):
    checkpoint = torch.load(model_filename)
    model.load_state_dict(checkpoint)
else:
    raise RuntimeError
print("Trained model weights loaded.")

########## Test ##########
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.functional.nll_loss
training_record = [["epoch", "train loss", "val loss", "train path IoU", "val path IoU", "epoch time"]]

weights = torch.Tensor(dset['train'].labelweights).to(device)


# eval
test_loss, test_acc = [], []
model.eval()
test_start_time = time.time()
with torch.no_grad():
    for batch_idx, batch in enumerate(dloader['test']):
        if (batch_idx+1) % 10 == 0:
            time_left = (time.time() - test_start_time) * (len(dloader['test']) / (batch_idx + 1) - 1) / 60
            print("Validating {0}/{1}, remaining time: {2} min".format(batch_idx+1, len(dloader['test']), int(time_left)))
            print("Curren validation loss: {0:.4f}".format(np.mean(test_loss)))
            # break
        raw_img_input, img_input, img_label, token = batch
        img_input, img_label = img_input.to(device), img_label.to(device)
        img_pred_logits = model(img_input)

        img_pred_logits = nn.functional.log_softmax(img_pred_logits, dim=1) # (b,2,224,224)
        img_pred_logits = img_pred_logits.permute(0,2,3,1).reshape(-1, 2) # (b*h*w, 2)
        img_label = img_label.reshape(-1).long()# (b*h*w,)
        loss = nn.functional.nll_loss(img_pred_logits, img_label, weights)
        test_loss.append(loss.item())
        img_pred_flattened = img_pred_logits.detach().to('cpu').data.max(1)[1].float().reshape(batch_size, -1)
        img_label_flattened = img_label.detach().to('cpu').reshape(batch_size, -1)
        path_IoU = ((img_pred_flattened*img_label_flattened).sum(1)/(((img_pred_flattened+img_label_flattened)>0).float().sum(1)+1e-8)).tolist()
        test_acc += path_IoU
test_loss = np.mean(test_loss)
test_acc = np.mean(test_acc)

epoch_time = time.time()-test_start_time
print("unet - test_loss: {0:.3f}, test_optimal_path_IoU: {1:.3f}, time: {2} sec"\
    .format(test_loss, test_acc, int(epoch_time)))