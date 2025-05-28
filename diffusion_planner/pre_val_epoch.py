import torch
from tqdm import tqdm
from torchmetrics import Metric
from validation.visual import batch_output_to_np_list, viz_predictions
import os

class ADE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.sum += torch.norm(pred - target, p=2, dim=-1).mean(dim=-1).sum()
        self.count += pred.size(0)

    def compute(self):
        return self.sum / self.count


class FDE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.sum += torch.norm(pred[:, -1] - target[:, -1], p=2, dim=-1).sum()
        self.count += pred.size(0)

    def compute(self):
        return self.sum / self.count


class MR(Metric):
    def __init__(self, miss_threshold: float = 2.0):
        super().__init__()
        self.miss_threshold = miss_threshold
        self.add_state("misses", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        miss = torch.norm(pred[:, -1] - target[:, -1], p=2, dim=-1) > self.miss_threshold
        self.misses += miss.sum()
        self.total += pred.size(0)

    def compute(self):
        return self.misses.float() / self.total


@torch.no_grad()
def validation_epoch(model, val_loader, device,show = False ):
    model.eval()

    ade_metric = ADE().to(device)
    fde_metric = FDE().to(device)
    mr_metric = MR().to(device)

    total_loss = 0.0
    total_batches = 0

    save_base_dir = "/root/traj-diffusion/visualization/heading-fixed_MLP"

    batch_idx = 0
    for batch in tqdm(val_loader, desc="Validation", unit="batch"):
        batch_idx = batch_idx + 1 

        batch = batch.to(device)
        y_hat, pi = model(batch)
 
        reg_mask = ~batch['padding_mask'][:, 20:]
        l2_norm = (torch.norm(y_hat[:, :, :, :2] - batch.y, dim=-1) * reg_mask).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(batch.num_nodes)]
        reg_loss = model.reg_loss(y_hat_best[reg_mask], batch.y[reg_mask])
        total_loss += reg_loss.item()
        total_batches += 1

        # Agent-based evaluation
        y_hat_agent = y_hat[:, batch['agent_index'], :, :2]
        y_agent = batch.y[batch['agent_index']]
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        best_mode_agent = fde_agent.argmin(dim=0)
        y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(batch.num_graphs)]

        ade_metric.update(y_hat_best_agent, y_agent)
        fde_metric.update(y_hat_best_agent, y_agent)
        mr_metric.update(y_hat_best_agent, y_agent)

    avg_loss = total_loss / total_batches
    ade = ade_metric.compute().item()
    fde = fde_metric.compute().item()
    mr = mr_metric.compute().item()
    
    print(f"✅ Validation - Loss: {avg_loss:.4f}, ADE: {ade:.4f}, FDE: {fde:.4f}, MR: {mr:.4f}")
    return ade, fde, mr
