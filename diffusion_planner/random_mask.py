# # import torch

# def random_mask_agent_history(inputs, min_keep=2, history_steps=20):
#     # 1. 随机生成保留历史帧数 H
#     B = inputs['x'].shape[0]  # N or batch size
#     H = torch.randint(low=min_keep, high=history_steps+1, size=(B,))  # shape [B], H in [2, 20]

#     # 2. 修改 agent 的 padding_mask
#     new_padding_mask = inputs['padding_mask'].clone()  # [N, 50]
#     for i in range(B):
#         new_padding_mask[i, :history_steps-H[i]] = True   # 前面mask掉
#         new_padding_mask[i, history_steps-H[i]:history_steps] = False  # 后面保留
#     # 如果只对agent，可能B=1，或者用agent_index处理

#     # 3. 重新计算 bos_mask
#     new_bos_mask = torch.zeros_like(inputs['bos_mask'])
#     new_bos_mask[:, 0] = ~new_padding_mask[:, 0]
#     new_bos_mask[:, 1:history_steps] = new_padding_mask[:, :history_steps-1] & ~new_padding_mask[:, 1:history_steps]

#     # 4. 重新计算 x
#     x = inputs['x'].clone()
#     positions = inputs['positions']
#     x[:, 1:history_steps] = torch.where(
#         (new_padding_mask[:, :history_steps-1] | new_padding_mask[:, 1:history_steps]).unsqueeze(-1),
#         torch.zeros_like(x[:, 1:history_steps]),
#         positions[:, 1:history_steps] - positions[:, :history_steps-1]
#     )
#     x[:, 0] = torch.zeros_like(x[:, 0])

#     # 更新 inputs 并返回
#     inputs['x'] = x
#     inputs['padding_mask'] = new_padding_mask
#     inputs['bos_mask'] = new_bos_mask

#     return inputs


import torch
import random

def random_mask_agent_history(inputs, min_keep=2, history_steps=20):
    # Clone original data
    x = inputs['x'].clone()
    positions = inputs['positions'].clone()
    new_padding_mask = inputs['padding_mask'].clone()
    new_bos_mask = inputs['bos_mask'].clone()

    agent_indices = inputs['agent_index']  # shape: [B]

    B = agent_indices.shape[0]

    # 1. Randomly generate number of history steps to keep for each agent
    H = torch.randint(low=min_keep, high=history_steps + 1, size=(B,), device=x.device)  # [B]
    
    
    # H, L_opt = generate_h_lopt_pairs(B,device=x.device)
    # H = torch.full((B,), 20,device=x.device)  # Creates a tensor of shape [B] filled with 20
    # L_opt = torch.full((B,), 20,device=x.device)

    for i in range(B):
        agent_id = agent_indices[i]

        # 2. Update padding mask for the agent only
        # padding mask here is for position
        new_padding_mask[agent_id, :history_steps - H[i]] = True
        new_padding_mask[agent_id, history_steps - H[i]:history_steps] = False

        # 3. Update bos mask for the agent
        new_bos_mask[agent_id, 0] = ~new_padding_mask[agent_id, 0]
        new_bos_mask[agent_id, 1:history_steps] = new_padding_mask[agent_id, :history_steps - 1] & ~new_padding_mask[agent_id, 1:history_steps]

        # 4. Update x for the agent
        x[agent_id, 0] = torch.zeros_like(x[agent_id, 0])
        x[agent_id, 1:history_steps] = torch.where(
            (new_padding_mask[agent_id, :history_steps - 1] | new_padding_mask[agent_id, 1:history_steps]).unsqueeze(-1),
            torch.zeros_like(x[agent_id, 1:history_steps]),
            positions[agent_id, 1:history_steps] - positions[agent_id, :history_steps - 1]
        )

    # Set updated values back into inputs
    inputs['x_copy']= inputs['x'].clone()
    inputs['x_random_mask'] = x
    inputs['x'] = x
    inputs['padding_mask'] = new_padding_mask
    inputs['bos_mask'] = new_bos_mask
    # Add the length after mask H into inputs
    inputs['H'] = H
    # inputs["L_opt"] = L_opt
    

    return inputs

def generate_h_lopt_pairs(batch_size, device):
    # Generate all valid (H, L_opt) pairs where L_opt > H
    valid_pairs = [(h, l) for h in range(2, 20) for l in range(2, 21) if l > h]
    
    # Uniformly sample batch_size pairs
    chosen_pairs = random.choices(valid_pairs, k=batch_size)

    # Convert to tensors on specified device
    H = torch.tensor([h for h, _ in chosen_pairs], dtype=torch.long, device=device)
    L_opt = torch.tensor([l for _, l in chosen_pairs], dtype=torch.long, device=device)

    return H, L_opt


def recalculate_masks(inputs, Lopt, history_steps=20):
    B = Lopt.shape[0]
    Lopt = Lopt.long()

    # [B, history_steps]
    t = torch.arange(history_steps, device=Lopt.device).unsqueeze(0).expand(B, -1)
    # NOTE: Lopt 
    keep_start = (history_steps - Lopt + 1).unsqueeze(1)

    new_padding_mask = t < keep_start  # True = mask

    bos_first = ~new_padding_mask[:, 0:1]  # [B,1]
    bos_rest = new_padding_mask[:, :-1] & ~new_padding_mask[:, 1:]  # [B,history_steps-1]
    new_bos_mask = torch.cat([bos_first, bos_rest], dim=1)  # [B, history_steps]

    # Clone original masks
    padding_mask = inputs['padding_mask'].clone()
    bos_mask = inputs['bos_mask'].clone()

    # Update only the first `history_steps` of agent_index rows
    padding_mask[inputs['agent_index'], :history_steps] = new_padding_mask
    bos_mask[inputs['agent_index'], :history_steps] = new_bos_mask

    # Assign back
    inputs['padding_mask'] = padding_mask
    inputs['bos_mask'] = bos_mask

    return inputs


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


def generate_displacement_mask( L_opt, T=20):
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