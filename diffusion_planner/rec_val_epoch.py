import torch
from tqdm import tqdm
from torchmetrics import Metric
from validation.visual import batch_output_to_np_list, viz_predictions
import os
import torch.nn.functional as F

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
    
    seen_ade_metric = ADE().to(device)
    unseen_ade_metric = ADE().to(device)

    loss_matrix = torch.zeros(19, 19, device=device)
    count_matrix = torch.zeros(19, 19, device=device)
    seen_loss_matrix = torch.zeros(19, 19, device=device)
    unseen_loss_matrix = torch.zeros(19, 19, device=device)
    seen_count_matrix = torch.zeros(19, 19, device=device)
    unseen_count_matrix = torch.zeros(19, 19, device=device)
    total_recon_loss = 0.0
    total_seen_loss = 0.0
    total_unseen_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation", unit="batch")):
        batch = batch.to(device)
        x0, decoder_outputs = model(batch)

        x_recon = x0
        x_gt = decoder_outputs["gt"]
        inputs_h = batch.H
        L_opt = batch["L_opt"]


        # visualization
        save_base_dir = "/root/traj-diffusion/visualization/recon_debug/diffusion_fixed"
        if show:
            scene_list = batch_output_to_np_list(
                batch, batch, x_recon, y_hat = None, 
                val_data_folder='/root/dataset/combined/val/data',only_recon = True
            )
            for scene_idx, scene in enumerate(scene_list):
                save_path = os.path.join(save_base_dir, f"batch_{batch_idx+1}", f"scene_{scene_idx}.png")
                viz_predictions(scene["input"], scene["output"] , scene["target"], scene["recon"],
                                scene["centerlines"], scene["city"], show=True, save_path=save_path,only_recon = True)

        
        

        stats = update_matrix(x_recon, x_gt, inputs_h, L_opt,seen_ade_metric,unseen_ade_metric)

        loss_matrix += stats["loss_matrix"]
        count_matrix += stats["count_matrix"]
        seen_loss_matrix += stats["seen_loss_matrix"]
        unseen_loss_matrix += stats["unseen_loss_matrix"]
        seen_count_matrix += stats["seen_count_matrix"]
        unseen_count_matrix += stats["unseen_count_matrix"]
        total_recon_loss += stats["total_recon_loss"]
        total_seen_loss += stats["total_seen_loss"]
        total_unseen_loss += stats["total_unseen_loss"]

    avg_total_recon_loss = total_recon_loss / count_matrix.sum()
    avg_total_seen_loss = total_seen_loss / seen_count_matrix.sum()
    avg_total_unseen_loss = total_unseen_loss / unseen_count_matrix.sum()

    loss_matrix[count_matrix > 0] /= count_matrix[count_matrix > 0]
    seen_loss_matrix[seen_count_matrix > 0] /= seen_count_matrix[seen_count_matrix > 0]

    unseen_loss_per_hidx = []
    for h_idx in range(19):
        # Get unseen losses and counts across l_idx for the current h_idx
        losses = unseen_loss_matrix[h_idx]
        counts = unseen_count_matrix[h_idx]

        # Avoid division by zero
        valid = counts > 0
        
        if valid.any():
            avg_loss = (losses[valid] / counts[valid]).mean().item()
        else:
            avg_loss = float('nan')  # or 0.0 if you prefer
        # import pdb; pdb.set_trace()
        unseen_loss_per_hidx.append(avg_loss)
    
    print("📊 Unseen Loss per H_idx (h = 2~20):")
    for i, val in enumerate(unseen_loss_per_hidx):
        print(f"h={i+2}: {val:.4f}")


    unseen_loss_matrix[unseen_count_matrix > 0] /= unseen_count_matrix[unseen_count_matrix > 0]


    seen_ade = seen_ade_metric.compute().item()
    unseen_ade = unseen_ade_metric.compute().item()
    print(f"\n✅ Final Recon Loss: {avg_total_recon_loss:.4f}, Seen: {avg_total_seen_loss:.4f}, Unseen: {avg_total_unseen_loss:.4f}")
    print(f"seen ade: {seen_ade}, unseen ade: {unseen_ade}")

    # TODO: calculate the unseen loss in each h_idx   matrix -> list of average losses in each h_idx
    


    return {
        "loss_matrix": loss_matrix,
        "count_matrix": count_matrix,
        "seen_loss_matrix": seen_loss_matrix,
        "unseen_loss_matrix": unseen_loss_matrix,
        "avg_total_recon_loss": avg_total_recon_loss,
        "avg_total_seen_loss": avg_total_seen_loss,
        "avg_total_unseen_loss": avg_total_unseen_loss
    }




# === Matrix Update Function ===
def update_matrix(x_recon, x_gt, inputs_h, L_opt,seen_ade_metric,unseen_ade_metric):
    B, T = x_gt.shape[:2]

    device = x_gt.device

    # Initialize batch-local metrics
    loss_matrix = torch.zeros(19, 19, device=device)
    count_matrix = torch.zeros(19, 19, device=device)
    seen_loss_matrix = torch.zeros(19, 19, device=device)
    unseen_loss_matrix = torch.zeros(19, 19, device=device)
    seen_count_matrix = torch.zeros(19, 19, device=device)
    unseen_count_matrix = torch.zeros(19, 19, device=device)
    total_recon_loss = 0.0
    total_seen_loss = 0.0
    total_unseen_loss = 0.0

    

    for i in range(B):
        h = int(inputs_h[i].item())
        l = int(L_opt[i].item())
        if 2 <= h <= 20 and 2 <= l <= 20:
            h_idx, l_idx = h - 2, l - 2
            valid_len = l - 1
            if valid_len > 0:
                loss = ((x_recon[i, -valid_len:, :] - x_gt[i, -valid_len:, :]) ** 2).mean()
                loss_matrix[h_idx, l_idx] += loss
                count_matrix[h_idx, l_idx] += 1
                total_recon_loss += loss.item()


            start_idx = T - l +1
            seen_len = min(h - 1, l - 1)
            unseen_len = max(0, l - h)

            if seen_len > 0:                
                x_gt_seen = x_gt[i, T - seen_len:T, :]
                x_recon_seen = x_recon[i, T - seen_len:T, :]
                loss_seen = ((x_recon_seen - x_gt_seen) ** 2).mean()
                # update ade
                seen_ade_metric.update(x_recon_seen,x_gt_seen)

                seen_loss_matrix[h_idx, l_idx] += loss_seen
                seen_count_matrix[h_idx, l_idx] += 1
                total_seen_loss += loss_seen.item()
                
                # if h == 2 or h == 19 or l == 20:
                # print(f"Edge case hit: h={h}, l={l}, seen_len={seen_len}, unseen_len={unseen_len}")
                # import pdb; pdb.set_trace()

            if unseen_len > 0:
                
                x_gt_unseen = x_gt[i, start_idx:start_idx + unseen_len, :]
                x_recon_unseen = x_recon[i, start_idx:start_idx + unseen_len, :]
                loss_unseen = ((x_recon_unseen - x_gt_unseen) ** 2).mean()

                # update ade
                unseen_ade_metric.update(x_recon_unseen,x_gt_unseen)

                unseen_loss_matrix[h_idx, l_idx] += loss_unseen
                unseen_count_matrix[h_idx, l_idx] += 1
                total_unseen_loss += loss_unseen.item()
                # if h == 2 or h == 19 or l == 20:
                # print(f"Edge case hit: h={h}, l={l}, seen_len={seen_len}, unseen_len={unseen_len}")
                # import pdb; pdb.set_trace()
                # print(f"Unseen loss h={h}, l={l}, start_idx={start_idx}, len={unseen_len}")
                # print(f"x_gt_unseen shape: {x_gt_unseen.shape}, x_recon_unseen shape: {x_recon_unseen.shape}")
                

    return {
        "loss_matrix": loss_matrix,
        "count_matrix": count_matrix,
        "seen_loss_matrix": seen_loss_matrix,
        "unseen_loss_matrix": unseen_loss_matrix,
        "seen_count_matrix": seen_count_matrix,
        "unseen_count_matrix": unseen_count_matrix,
        "total_recon_loss": total_recon_loss,
        "total_seen_loss": total_seen_loss,
        "total_unseen_loss": total_unseen_loss
    }
