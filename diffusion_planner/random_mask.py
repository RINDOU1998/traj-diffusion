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

    for i in range(B):
        agent_id = agent_indices[i]

        # 2. Update padding mask for the agent only
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
    inputs['x'] = x
    inputs['padding_mask'] = new_padding_mask
    inputs['bos_mask'] = new_bos_mask
    # Add the length after mask H into inputs
    inputs['H'] = H

    return inputs

def recalculate_masks(inputs, Lopt,history_steps=20):
    new_padding_mask = inputs['padding_mask'].clone()
    new_bos_mask = inputs['bos_mask'].clone()
    agent_indices = inputs['agent_index']  # shape: [B]
    B = agent_indices.shape[0]
    for i in range(B):
        agent_id = agent_indices[i]
        # 2. Update padding mask for the agent only
        new_padding_mask[agent_id, :history_steps - Lopt[i]] = True
        new_padding_mask[agent_id, history_steps - Lopt[i]:history_steps] = False

        # 3. Update bos mask for the agent
        new_bos_mask[agent_id, 0] = ~new_padding_mask[agent_id, 0]
        new_bos_mask[agent_id, 1:history_steps] = new_padding_mask[agent_id, :history_steps - 1] & ~new_padding_mask[agent_id, 1:history_steps]
    
    inputs['padding_mask'] = new_padding_mask
    inputs['bos_mask'] = new_bos_mask
    
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