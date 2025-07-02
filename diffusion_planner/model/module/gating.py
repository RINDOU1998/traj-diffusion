import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusion_planner.utils.utils import init_weights

class LoptDecoder(nn.Module):
    def __init__(self, embed_dim: int , T_max=20):
        super().__init__()
        self.attn_score = nn.Linear(embed_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, T_max - 1),
            nn.LayerNorm(T_max - 1)  
        )
        self.apply(init_weights)

    def forward(self, x_encoded, current_epoch = 64):  # [B, T, D]
        # cosine annealing tau
        tau = self.cosine_annealing_tau(current_epoch)
        ################# position-wise embed decoder version #################
        # attn_weights = torch.softmax(self.attn_score(x_encoded), dim=1)  # [B, T, 1]
        # pooled = torch.sum(attn_weights * x_encoded, dim=1)  # [B, D]
        # length_logits = self.mlp(pooled)  # [B, T-1]
        # import pdb; pdb.set_trace()
        ##################################################################
        


        ################## global embed decoder version ##################
        # discrete length distribution                   
        length_logits =  self.mlp(x_encoded)
        L_opt_one_hot, L_opt = self.sample_Lopt(length_logits,tau = tau)  # [B, T-1], [B]
        mask = self.generate_displacement_mask(L_opt)
        # import pdb; pdb.set_trace()
        return L_opt, mask, length_logits
    
    def sample_Lopt(self,logits: torch.Tensor, tau: float = 0.1):
        """
        Args:
            logits: [B, T-1]  # MLP output for length options
            tau: Temperature for Gumbel-Softmax
        Returns:
            L_opt_one_hot: [B, T-1] (differentiable one-hot vector)
            L_opt: [B] (expected integer lengths)
        """
        # Step 1: differentiable one-hot
        L_opt_one_hot = F.gumbel_softmax(logits, tau=tau, hard=True)  # [B, T-1]

        # L_opt_one_hot, debug_info = self.debug_gumbel_softmax(logits, tau=tau, hard=True)

     
        

        # Step 2: construct length options [ 2, ..., T]
        length_range = torch.arange(2, logits.size(1) + 2, device=logits.device).float()  # [T-1]

        # Step 3: weighted sum to get L_opt
        L_opt = torch.sum(L_opt_one_hot * length_range.unsqueeze(0), dim=1)  # [B]

        
        return L_opt_one_hot, L_opt  # L_opt is float, you can .floor() or .long() as needed

    def debug_gumbel_softmax(self, logits, tau=1.0, hard=True):
        """
        可调试的Gumbel-Softmax实现
        返回: (采样结果, 调试信息字典)
        """
        debug_info = {}
        
        # 记录原始logits
        debug_info['input_logits'] = logits.detach()
        
        # 1. 计算softmax概率
        probs = F.softmax(logits / tau, dim=-1)
        debug_info['softmax_probs'] = probs.detach()
        
        # 2. 添加Gumbel噪声
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits, device=logits.device)))
        noisy_logits = (logits + gumbel_noise) / tau
        debug_info['noisy_logits'] = noisy_logits.detach()
        
        # 3. 计算带噪声的softmax
        noisy_probs = F.softmax(noisy_logits, dim=-1)
        debug_info['noisy_probs'] = noisy_probs.detach()
        
        # 4. 直通式(hard)采样
        if hard:
            _, max_indices = torch.max(noisy_probs, dim=-1, keepdim=True)
            hard_samples = torch.zeros_like(noisy_probs, device=logits.device).scatter_(-1, max_indices, 1.0)
            debug_info['hard_samples'] = hard_samples.detach()
            # import pdb; pdb.set_trace()
            return hard_samples, debug_info
        
        
        return noisy_probs, debug_info

    def generate_displacement_mask(self, L_opt, T=20):
        """
        Args:
            L_opt: [B], values in [2, T]
        Returns:
            mask: [B, T] boolean mask. True = masked, False = keep
        """
        B = L_opt.shape[0]
        idx = torch.arange(T, device=L_opt.device).unsqueeze(0).expand(B, -1)  # [B, T]
        keep_start = (T - L_opt).unsqueeze(1)  # [B, 1]
        # Mask positions where idx <= keep_start
        mask = idx <= keep_start  # [B, T]
        return mask
    
    def cosine_annealing_tau(self, step, total_steps=64, max_tau=100.0, min_tau=0.1):
        step = min(step, total_steps)  # clamp to total_steps
        cosine = math.cos(math.pi * step / total_steps)
        tau = min_tau + 0.5 * (max_tau - min_tau) * (1 + cosine)
        return tau


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