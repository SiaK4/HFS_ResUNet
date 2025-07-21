import os
import json
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import time

from Dataset.dataset import HDF5ConcatDataset
from Dataset.dataset import TempInputDataset
from utils.save_model import save_checkpoint
from Models.ResUnet_HFS import ResUNet
from lion_pytorch import Lion

torch.manual_seed(123)
np.random.seed(123)
torch.cuda.empty_cache()
batch_size = 4

# Loading the data
data_dir = './BubbleML/PoolBoiling-SubCooled-FC72-2D'
training_files = ['Twall-79.hdf5','Twall-81.hdf5','Twall-85.hdf5','Twall-90.hdf5','Twall-100.hdf5','Twall-103.hdf5','Twall-106.hdf5','Twall-110.hdf5']
training_files = [os.path.join(data_dir,file) for file in training_files]
validation_files = ['Twall-95.hdf5','Twall-98.hdf5']
validation_files =[os.path.join(data_dir,file) for file in validation_files]

val1 = TempInputDataset(validation_files[0], steady_time=30, use_coords=False, time_window=5, future_window=5)
val2 = TempInputDataset(validation_files[0], steady_time=30, use_coords=False, time_window=5, future_window=5)
train_dataset = HDF5ConcatDataset(TempInputDataset(file, steady_time=30, use_coords=False, time_window=5, future_window=5) for file in training_files)
print("Number of training data:",len(train_dataset))


#Normalize training dataset
global_max_temp = train_dataset.normalize_temp_()
global_max_vel = train_dataset.normalize_vel_()
print("global max temperature:",global_max_temp)
print("global max velocity:",global_max_vel)

global_min_temp = train_dataset.find_min_temp()
print("global minimum temperature:",global_min_temp)

#Normalize validation dataset using the scale from training dataset
val1.normalize_temp_(global_max_temp)
val1.normalize_vel_(global_max_vel)
val2.normalize_temp_(global_max_temp)
val2.normalize_vel_(global_max_vel)
val_dataset = HDF5ConcatDataset([val1,val2])
print("Number of validation data:",len(val_dataset))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def train_step(model,x,y):
    y_pred = model(x)
    loss = torch.mean((y_pred-y)**2)
    return loss

def warmup_lr(optimizer, scheduler1, scheduler2, current_step, warmup_steps, initial_lr,target_lr):
    if current_step <= warmup_steps:
        lr = initial_lr + (target_lr - initial_lr)*(current_step/warmup_steps)
        optimizer.param_groups[0]['lr'] = lr
        scheduler1.base_lrs = [group['lr'] for group in optimizer.param_groups]
        scheduler2.base_lrs = [group['lr'] for group in optimizer.param_groups]

def train(model, epoch_number, learning_rate, target_lr,checkpoint_name, display_every=10, checkpoint_interval=50,warmup_steps=10):
    best_loss = float('inf')
    train_loss = []
    val_loss = []
    model.to(device)

    # optimizer = optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=0.01)
    optimizer = Lion(model.parameters(), lr=learning_rate, weight_decay = 0.06)
    scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.9)
    scheduler2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5, T_mult=1, eta_min=8e-06)
    epoch_time = 0

    for epoch in range(epoch_number):
        ## Optional: call the learning rate warmup
        # warmup_lr(optimizer,scheduler1,scheduler2,epoch,warmup_steps,initial_lr=learning_rate,target_lr=target_lr)
        epoch_start = time.time()
        batch_loss = []
        for i,data in enumerate(train_loader):
            x_train, y_train = ResUNet.set_input(data)
            loss = train_step(model, x_train, y_train)
            optimizer.zero_grad()
            loss.backward()
            #Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
            optimizer.step()
            
            batch_loss.append(loss.item())
        time_for_epoch = time.time() - epoch_start
        epoch_time+=time_for_epoch
        train_loss.append(np.mean(batch_loss))

        with torch.no_grad():
            batch_val_loss = []
            for j,val_data in enumerate(val_loader):
                x_val, y_val = ResUNet.set_input(val_data)
                loss_val = train_step(model, x_val, y_val)
                batch_val_loss.append(loss_val.item())
            val_loss.append(np.mean(batch_val_loss))

        if (epoch+1)%checkpoint_interval == 0:
            best_loss = save_checkpoint(model, epoch, loss_val, best_loss, checkpoint_name=checkpoint_name)
        
        lr_old = optimizer.param_groups[0]['lr']
        if (epoch+1)>=900 and (epoch+1)<=1300:
            if (epoch+1) ==1100:
                scheduler1.base_lrs = [group['lr'] for group in optimizer.param_groups]
            scheduler1.step()
        lr_new = optimizer.param_groups[0]['lr']
        if lr_old != lr_new:
            print(f"Learning rate has changed {lr_old:.8f}--->{lr_new:.8f} at epoch {epoch+1}")
        
        if ((epoch+1)%display_every==0) or epoch==0:
            print(f"Training loss is {(np.mean(batch_loss)).item()} at epoch {epoch+1} <><><><><> Validation loss is {(np.mean(batch_val_loss)).item()}")
            print("epoch time:", time_for_epoch)

        ### Save the losses every 20 epochs in case training is left incomplete
        if (epoch+1)%20 == 0:
            with open('train_loss.json','w') as file:
                json.dump(train_loss, file)
            with open('val_loss.json','w') as file2:
                json.dump(val_loss, file2)
    
    print("Average per epoch time:",epoch_time/epoch_number)
    
    #Save last epoch
    best_loss = float('inf')
    save_checkpoint(model, epoch, loss_val, best_loss=best_loss, checkpoint_name=checkpoint_name)
    
# model = ResUNet(in_c=25, out_c=5,features=[32,64,64,128,128], bottleneck_feature=256)  ## Without HFS
# model = UNet(in_c=25, out_c=5, n_layers = 4) ## Unet for benchmarking
model = ResUNet(in_c = 25, out_c = 5, features=[32,64,64,128,128], bottleneck_feature=256, patch_size= [16,8,4,2,1])

if torch.cuda.is_available():
    device = torch.device('cuda')

train(model, epoch_number=1400, learning_rate=8e-4,target_lr=8e-4,checkpoint_name='check_HFS',display_every=1,checkpoint_interval=20,warmup_steps=10)
