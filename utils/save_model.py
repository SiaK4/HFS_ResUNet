import torch
import os

def save_checkpoint(model,epoch,loss,best_loss,checkpoint_name):
    checkpoint_dir = f"checkpoints_{checkpoint_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    if loss < best_loss:
        print(f"Loss has decreased ({best_loss:.4f}-->{loss:.4f}). Saving the checkpoint at epoch {epoch+1}")
        torch.save(model.state_dict(),os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pth"))

        return loss
    return best_loss