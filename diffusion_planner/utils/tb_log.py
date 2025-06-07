import os
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torch

import wandb
import matplotlib.pyplot as plt
import seaborn as sns

def save_matrix_heatmap(matrix, title, save_path):
    """
    Save a heatmap of the given matrix with annotations and good readability.
    
    Args:
        matrix (torch.Tensor): 2D tensor of shape [19, 19]
        title (str): Plot title
        save_path (str): Full path to save the figure
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    matrix = matrix.detach().cpu().numpy()

    plt.figure(figsize=(12, 10))  # ✅ Bigger figure
    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        xticklabels=range(2, 21),
        yticklabels=range(2, 21),
        annot_kws={"size": 8}  # ✅ Smaller annotation text
    )
    ax.set_xlabel("L_opt", fontsize=12)
    ax.set_ylabel("H", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class TensorBoardLogger():
    def __init__(self, run_name, notes, args, wandb_resume_id, save_path, rank=0):
        """
        project_name (str): wandb project name
        config: dict or argparser
        """              
        self.args = args
        self.writer = None
        self.id = None
        
        if rank == 0:
            os.environ["WANDB_MODE"] = "online" if args.use_wandb else "offline"

            wandb_writer = wandb.init(project='Diffusion-Planner', 
                name=run_name, 
                notes=notes,
                resume="allow",
                id = wandb_resume_id,
                sync_tensorboard=True,
                dir=f'{save_path}')
            wandb.config.update(args)
            self.id = wandb_writer.id
            
            self.writer = SummaryWriter(log_dir=f'{save_path}/tb')
    
    def log_metrics(self, metrics: dict, step: int):
       """
       metrics (dict):
       step (int, optional): epoch or step
       """
       if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def finish(self):
       if self.writer is not None:
            self.writer.close()

    def log_matrix_image(self, matrix: torch.Tensor, tag: str, step: int, normalize=True):
            """
            Logs a 2D tensor (e.g., loss matrix) as an image to TensorBoard.
            
            Args:
                matrix (torch.Tensor): 2D tensor of shape [H, W]
                tag (str): Name for this image (e.g., 'recon/loss_matrix')
                step (int): Global step (usually epoch)
                normalize (bool): Whether to normalize matrix to [0, 1]
            """
            if self.writer is None:
                return

            matrix = matrix.clone().detach().float().cpu()  # Ensure it's on CPU
            if normalize:
                matrix -= matrix.min()
                if matrix.max() > 0:
                    matrix /= matrix.max()

            # Convert to [1, H, W] for grayscale image
            img = matrix.unsqueeze(0)

            # Optional: upscale for visibility
            img = transforms.Resize((190, 190))(img)

            self.writer.add_image(tag, img, global_step=step)