import os
import json
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import time

from Dataset.kolmogorov_data import CustomDataset
from utils.save_model_kolmo import save_checkpoint
from Models.kolmogorov.ResUnet_HFS_encoder import ResUNet

from lion_pytorch import Lion
torch.manual_seed(1234)
np.random.seed(1234)

torch.cuda.empty_cache()
batch_size = 8

# Loading the data
data_dir = './kolmogorov_data.npz'
train_dataset = CustomDataset(data_dir=data_dir, mode='train')

train_min, train_max = train_dataset.get_min_max()
print("train_min:",train_min)
print("train_max:",train_max)
# Train data is normalized

#Normalize validation dataset using the scale from training dataset
val_dataset = CustomDataset(data_dir=data_dir, mode='val',train_min=train_min, train_max=train_max)

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
    optimizer = Lion(model.parameters(), lr=learning_rate, weight_decay = 0.01)
    scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    scheduler2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10, T_mult=1, eta_min=5e-06)
    epoch_time = 0

    for epoch in range(epoch_number):
        ## Optionally call warmup learning rate function
        # warmup_lr(optimizer,scheduler1,scheduler2,epoch,warmup_steps,initial_lr=learning_rate,target_lr=target_lr)
        epoch_start = time.time()
        batch_loss = []
        for i,data in enumerate(train_loader):
            x_train, y_train = ResUNet.set_input2(data)
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
                x_val, y_val = ResUNet.set_input2(val_data)
                loss_val = train_step(model, x_val, y_val)
                batch_val_loss.append(loss_val.item())
            val_loss.append(np.mean(batch_val_loss))

        if (epoch+1)%checkpoint_interval == 0:
            best_loss = save_checkpoint(model, epoch, loss_val, best_loss, checkpoint_name=checkpoint_name)

        ### Interval savings
        if (epoch+1)%51 == 0 or (epoch+1)%101 == 0 or (epoch+1)%151==0 or (epoch+1)%201==0 or (epoch+1)%251== 0:
            best_loss = float('inf')
            best_loss = save_checkpoint(model, epoch, loss_val, best_loss, checkpoint_name=checkpoint_name)
        
        lr_old = optimizer.param_groups[0]['lr']
        if (epoch+1)>=120 and (epoch+1)<=200:
            if (epoch+1) ==120:
                scheduler1.base_lrs = [group['lr'] for group in optimizer.param_groups]
            scheduler1.step()
        elif (epoch+1)>250:
            if (epoch+1)== 251:
                scheduler2.base_lrs = [group['lr'] for group in optimizer.param_groups]
            scheduler2.step()
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
    
    ### Print out the average epoch time
    print("Average per epoch time:",epoch_time/epoch_number)
    
    #Save last epoch
    best_loss = float('inf')
    save_checkpoint(model, epoch, loss_val, best_loss=best_loss, checkpoint_name=checkpoint_name)
    
model = ResUNet(in_c = 20,out_c = 5, features = [32,64,64,128,128], bottleneck_feature=256)

if torch.cuda.is_available():
    device = torch.device('cuda')

save_dir = 'checkpoints'   ## Add your directory to save the checkpoints
train(model, epoch_number=1000, learning_rate=8e-4,target_lr=8e-4,checkpoint_name=save_dir+'kolmo_HFS',display_every=10,checkpoint_interval=5,warmup_steps=10)
