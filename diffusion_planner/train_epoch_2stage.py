# diffusion_planner/train_epoch.py
from tqdm import tqdm
import torch
from torch import nn
from diffusion_planner.utils.train_utils import get_epoch_mean_loss
from diffusion_planner.utils import ddp
from diffusion_planner.loss import diffusion_loss_func
import math

def train_epoch(data_loader, model, optimizer, args,current_epoch, aug=None):
    model.train()
    if args.ddp:
        torch.cuda.synchronize()

    epoch_losses = []
    pbar = tqdm(data_loader, desc="Training", unit="batch")
    for batch in pbar: 
        batch = batch.to(args.device)
        optimizer.zero_grad()

        # returns a dict with keys:
        #   reconstruction_loss, regression_loss, classification_loss
        losses, _ = diffusion_loss_func(
            model, batch,current_epoch, args.state_normalizer, {}, args.diffusion_model_type
        )

        # pick the combination
        if model.stage == "recon":
            total =  losses["reconstruction_loss"]
        elif model.stage == "pred":
            total = losses["regression_loss"] + losses["classification_loss"]
        else:  # joint
            total = (losses["reconstruction_loss"]
                     + losses["regression_loss"]
                     + losses["classification_loss"]
                     + losses['entropy']* cosine_decay_lambda(current_epoch)
                     )
            

        losses["loss"] = total
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if args.ddp:
            torch.cuda.synchronize()

        epoch_losses.append(losses)
        pbar.set_postfix(loss=f"{total.item():.4f}")

    epoch_mean = get_epoch_mean_loss(epoch_losses)
    if args.ddp:
        epoch_mean = ddp.reduce_and_average_losses(epoch_mean,
                                                   torch.device(args.device))
    if ddp.get_rank() == 0:
        print(f"→ epoch mean loss: {epoch_mean['loss']:.4f}")
    return epoch_mean, epoch_mean["loss"]


def cosine_decay_lambda(epoch, total_epochs = 64, initial_lambda=0.1, final_lambda=0.0):
    """Decay lambda from initial_lambda to final_lambda using cosine annealing."""
    progress = min(epoch / total_epochs, 1.0)
    decay = 0.5 * (1 + math.cos(math.pi * progress))
    return final_lambda + (initial_lambda - final_lambda) * decay
