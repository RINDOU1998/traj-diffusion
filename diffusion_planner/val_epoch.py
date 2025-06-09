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
def validation_epoch(model, val_loader, device, show=False):
    model.eval()

    ade_metric = ADE().to(device)
    fde_metric = FDE().to(device)
    mr_metric = MR().to(device)

    total_loss = 0.0
    total_batches = 0
    save_base_dir = "/root/traj-diffusion/visualization/heading-fixed_MLP"

    # === Recon Metrics Accumulator ===
    loss_matrix = torch.zeros(19, 19, device=device)
    count_matrix = torch.zeros(19, 19, device=device)
    seen_loss_matrix = torch.zeros(19, 19, device=device)
    unseen_loss_matrix = torch.zeros(19, 19, device=device)
    seen_count_matrix = torch.zeros(19, 19, device=device)
    unseen_count_matrix = torch.zeros(19, 19, device=device)
    total_seen_loss = 0.0
    total_unseen_loss = 0.0
    total_recon_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation", unit="batch")):
        batch = batch.to(device)
        _, diffusion_output, y_hat, pi, batch_inputs = model(batch)

        # visualization
        if show:
            scene_list = batch_output_to_np_list(
                batch, batch_inputs, diffusion_output['x0'], y_hat, 
                val_data_folder='/root/dataset/val/data'
            )
            for scene_idx, scene in enumerate(scene_list):
                save_path = os.path.join(save_base_dir, f"batch_{batch_idx+1}", f"scene_{scene_idx}.png")
                viz_predictions(scene["input"], scene["output"], scene["target"], scene["recon"],
                                scene["centerlines"], scene["city"], show=True, save_path=save_path)


        # === Recon Loss Collection ===
        x_recon = batch_inputs.x[batch_inputs.agent_index]
        x_gt = batch_inputs.x_gt
        inputs_h = batch_inputs.H
        L_opt = batch_inputs['L_opt']
        B, T = x_gt.shape[:2]

        for i in range(B):
            h = int(inputs_h[i].item())
            l = int(L_opt[i].item())
            if 2 <= h <= 20 and 2 <= l <= 20:
                valid_len = l - 1
                if valid_len <= 0:
                    continue
                loss = ((x_recon[i, -valid_len:, :] - x_gt[i, -valid_len:, :]) ** 2).mean()
                loss_matrix[h - 2, l - 2] += loss
                count_matrix[h - 2, l - 2] += 1
                total_recon_loss += loss.item()

        for i in range(B):
            h = int(inputs_h[i].item())
            l = int(L_opt[i].item())
            if 2 <= h <= 20 and 2 <= l <= 20:
                h_idx, l_idx = h - 2, l - 2

                # NOTE check here
                start_idx = T - l + 1
                seen_len = min(h - 1, l - 1)
                unseen_len = max(0, l - h)

                if seen_len > 0:
                    x_gt_seen = x_gt[i, T - seen_len:T, :]
                    x_recon_seen = x_recon[i, T - seen_len:T, :]
                    loss_seen = ((x_recon_seen - x_gt_seen) ** 2).mean()
                    seen_loss_matrix[h_idx, l_idx] += loss_seen
                    seen_count_matrix[h_idx, l_idx] += 1
                    total_seen_loss += loss_seen.item()

                if unseen_len > 0:
                    x_gt_unseen = x_gt[i, start_idx:start_idx + unseen_len, :]
                    x_recon_unseen = x_recon[i, start_idx:start_idx + unseen_len, :]
                    loss_unseen = ((x_recon_unseen - x_gt_unseen) ** 2).mean()
                    unseen_loss_matrix[h_idx, l_idx] += loss_unseen
                    unseen_count_matrix[h_idx, l_idx] += 1
                    total_unseen_loss += loss_unseen.item()

        # === Mode Regression Loss ===
        reg_mask = ~batch['padding_mask'][:, 20:]
        l2_norm = (torch.norm(y_hat[:, :, :, :2] - batch.y, dim=-1) * reg_mask).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(batch.num_nodes)]
        reg_loss = model.reg_loss(y_hat_best[reg_mask], batch.y[reg_mask])
        total_loss += reg_loss.item()
        total_batches += 1

        # === Trajectory-based Evaluation ===
        y_hat_agent = y_hat[:, batch['agent_index'], :, :2]
        y_agent = batch.y[batch['agent_index']]
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], dim=-1)
        best_mode_agent = fde_agent.argmin(dim=0)
        y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(batch.num_graphs)]

        ade_metric.update(y_hat_best_agent, y_agent)
        fde_metric.update(y_hat_best_agent, y_agent)
        mr_metric.update(y_hat_best_agent, y_agent)

    # === Average Metrics ===
    avg_loss = total_loss / total_batches
    total_recon_loss /= count_matrix.sum()
    avg_total_seen_loss = total_seen_loss / seen_count_matrix.sum()
    avg_total_unseen_loss = total_unseen_loss / unseen_count_matrix.sum()


    loss_matrix[count_matrix > 0] /= count_matrix[count_matrix > 0]
    seen_loss_matrix[seen_count_matrix > 0] /= seen_count_matrix[seen_count_matrix > 0]
    unseen_loss_matrix[unseen_count_matrix > 0] /= unseen_count_matrix[unseen_count_matrix > 0]
    
    recon_loss_matrix = {
        "loss_matrix": loss_matrix,
        "count_matrix": count_matrix,
        "seen_loss_matrix": seen_loss_matrix,
        "unseen_loss_matrix": unseen_loss_matrix
    }

    ade = ade_metric.compute().item()
    fde = fde_metric.compute().item()
    mr = mr_metric.compute().item()

    print(f"✅ Validation - Loss: {avg_loss:.4f}, ADE: {ade:.4f}, FDE: {fde:.4f}, MR: {mr:.4f}, "
          f"Recon: {total_recon_loss:.4f}, Seen: {avg_total_seen_loss:.4f}, Unseen: {avg_total_unseen_loss:.4f}")

    return ade, fde, mr, avg_loss, total_recon_loss, avg_total_seen_loss, avg_total_unseen_loss, recon_loss_matrix
