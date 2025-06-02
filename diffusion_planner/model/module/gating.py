import torch
import torch.nn as nn

class gating(nn.Module):
    def __init__(self, embed_dim: int ,hidden_size = 256,  T_max = 20):
        super(gating, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.T_max = T_max
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        L_opt = self.proj(cls_token) *  self.T_max # [B]
        mask = self.generate_mask(L_opt)
        return L_opt.squeeze(1), mask.squeeze(1)
    

    def generate_mask(self, L_opt, temperature=0.1):
        """
        Args:
            L_opt: [B], optimal length (e.g., 5.3).
            temperature: Controls steepness (lower = sharper cutoff).
        Returns:
            mask: [B, T_max], differentiable w.r.t L_opt.
        """
        t = torch.arange(self.T_max, device=L_opt.device).float()  # [T_max]
        mask = torch.sigmoid((L_opt.unsqueeze(1) - t) / temperature)  # [B, T_max]
        return mask



def mask_x_gt_by_lopt(x_gt: torch.Tensor, L_opt: torch.Tensor) -> torch.Tensor:
    """
    Mask the x_gt tensor according to optimized length L_opt.

    Args:
        x_gt: [B, T, 2] trajectory displacement ground truth
        L_opt: [B,] optimized length (float), within [2, T]

    Returns:
        masked_x_gt: [B, T, 2] with first (T - floor(L_opt)) values zeroed
    """
    B, T, D = x_gt.shape
    device = x_gt.device

    # 1. Get integer length to keep (clip to [2, T])
    L_keep = L_opt.floor().clamp(min=2, max=T).long()  # [B]

    # 2. Create a time index matrix [T] → [1, T]
    time_idx = torch.arange(T, device=device).unsqueeze(0)  # [1, T]

    # 3. Create mask for each batch
    # keep positions where time_idx >= (T - L_keep[i])
    keep_mask = time_idx > (T - L_keep).unsqueeze(1)  # [B, T]

    # 4. Expand to [B, T, 2] and apply
    keep_mask = keep_mask.unsqueeze(-1).expand(-1, -1, D)  # [B, T, 2]
    masked_x_gt = torch.where(keep_mask, x_gt, torch.zeros_like(x_gt))

    return masked_x_gt