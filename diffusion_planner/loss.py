from typing import Any, Callable, Dict, List, Tuple
import torch
import torch.nn as nn
from diffusion_planner.utils.utils import TemporalData
from diffusion_planner.utils.normalizer import StateNormalizer

"""
Training phase
    input : 
        sampled trajectories [B,1,T,V] with current without noise
        diffusion time[B,]
    output
        score : [B,1,T,V] with noise


Args:
    model (nn.Module): The neural network model used for diffusion-based prediction.
    inputs (Dict[str, torch.Tensor]): A dictionary containing input tensors, including the current state of the ego 
        vehicle and neighboring agents.
    marginal_prob (Callable[[torch.Tensor], torch.Tensor]): A callable function that computes the mean and standard 
        deviation of the marginal probability distribution at a given diffusion time.
    norm (StateNormalizer): A normalizer object used to normalize the ground truth future states.
    loss (Dict[str, Any]): A dictionary to store the computed loss values.
    model_type (str): The type of model output, either "score" or "x_start", determining the loss computation method.
    eps (float, optional): A small constant to ensure numerical stability in diffusion time sampling. Defaults to 1e-3.
Returns:
    Tuple[Dict[str, Any], Any]: A tuple containing the updated loss dictionary and the decoder output from the model.
Raises:
    AssertionError: If the computed loss contains NaN values.
"""
def diffusion_loss_func(
    model: nn.Module,
    inputs: TemporalData,
    #futures: Tuple[torch.Tensor, torch.Tensor],
    norm: StateNormalizer,
    loss: Dict[str, Any],

    model_type: str,
    eps: float = 1e-3,
):   
    
    #NOTE prepare sampled trajectories and diffusion time, now move to decoder part  
    # ego_future, neighbors_future, neighbor_future_mask = futures
    # neighbors_future_valid = ~neighbor_future_mask # [B, P, V]

    # B, Pn, T, _ = neighbors_future.shape
    # ego_current, neighbors_current = inputs["ego_current_state"][:, :4], inputs["neighbor_agents_past"][:, :Pn, -1, :4]
    # neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    # neighbor_mask = torch.concat((neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1)

    # gt_future = torch.cat([ego_future[:, None, :, :], neighbors_future[..., :]], dim=1) # [B, P = 1 + 1 + neighbor, T, 4]
    # current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1) # [B, P, 4]

    # P = gt_future.shape[1]
    # t = torch.rand(B, device=gt_future.device) * (1 - eps) + eps # [B,]
    # z = torch.randn_like(gt_future, device=gt_future.device) # [B, P, T, 4]
    
    # all_gt = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)
    # all_gt[:, 1:][neighbor_mask] = 0.0

    # mean, std = marginal_prob(all_gt[..., 1:, :], t)
    # std = std.view(-1, *([1] * (len(all_gt[..., 1:, :].shape)-1)))

    # xT = mean + std * z
    # xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)
    
    # merged_inputs = {
    #     **inputs,
    #     "sampled_trajectories": xT,
    #     "diffusion_time": t,
    # }

    _, decoder_output = model(inputs) # [B, P, 1 + T, 4]
    # TODO continue after decoder complete

    score = decoder_output["score"][:, :, 1:, :] # [B, P, T, 4]

    if model_type == "score":
        dpm_loss = torch.sum((score * std + z)**2, dim=-1)
    elif model_type == "x_start":
        dpm_loss = torch.sum((score - all_gt[:, :, 1:, :])**2, dim=-1)
    
    masked_prediction_loss = dpm_loss[:, 1:, :][neighbors_future_valid]

    if masked_prediction_loss.numel() > 0:
        loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()
    else:
        loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=masked_prediction_loss.device)

    loss["ego_planning_loss"] = dpm_loss[:, 0, :].mean()

    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

    return loss, decoder_output