# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data


class TemporalData(Data):

    def __init__(self,
                 x: Optional[torch.Tensor] = None,
                 positions: Optional[torch.Tensor] = None,
                 edge_index: Optional[torch.Tensor] = None,
                 edge_attrs: Optional[List[torch.Tensor]] = None,
                 y: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 padding_mask: Optional[torch.Tensor] = None,
                 bos_mask: Optional[torch.Tensor] = None,
                 rotate_angles: Optional[torch.Tensor] = None,
                 lane_vectors: Optional[torch.Tensor] = None,
                 is_intersections: Optional[torch.Tensor] = None,
                 turn_directions: Optional[torch.Tensor] = None,
                 traffic_controls: Optional[torch.Tensor] = None,
                 lane_actor_index: Optional[torch.Tensor] = None,
                 lane_actor_vectors: Optional[torch.Tensor] = None,
                 seq_id: Optional[int] = None,
                 **kwargs) -> None:
        if x is None:
            super(TemporalData, self).__init__()
            return
        super(TemporalData, self).__init__(x=x, positions=positions, edge_index=edge_index, y=y, num_nodes=num_nodes,
                                           padding_mask=padding_mask, bos_mask=bos_mask, rotate_angles=rotate_angles,
                                           lane_vectors=lane_vectors, is_intersections=is_intersections,
                                           turn_directions=turn_directions, traffic_controls=traffic_controls,
                                           lane_actor_index=lane_actor_index, lane_actor_vectors=lane_actor_vectors,
                                           seq_id=seq_id, **kwargs)
        if edge_attrs is not None:
            for t in range(self.x.size(1)):
                self[f'edge_attr_{t}'] = edge_attrs[t]

    def __inc__(self, key, value,store=None):
        if key == 'lane_actor_index':
            return torch.tensor([[self['lane_vectors'].size(0)], [self.num_nodes]])
        else:
            return super().__inc__(key, value, store)


class DistanceDropEdge(object):

    def __init__(self, max_distance: Optional[float] = None) -> None:
        self.max_distance = max_distance

    def __call__(self,
                 edge_index: torch.Tensor,
                 edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.max_distance is None:
            return edge_index, edge_attr
        row, col = edge_index
        mask = torch.norm(edge_attr, p=2, dim=-1) < self.max_distance
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        edge_attr = edge_attr[mask]
        return edge_index, edge_attr


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)


def reconstruct_absolute_position_from_last_frame(x0, inputs, reverse_rotation = False):
    """
    Reconstruct absolute global positions of x0 (agent trajectory) by anchoring the last frame (t=19).

    Args:
        x0: torch.Tensor of shape [B,1, 20, 2], displacement-encoded trajectory relative to previous step
        inputs: TemporalData batch with fields:
            - inputs['positions']: original AV-centric absolute pos, shape [N, 50, 2]
            - inputs['agent_index']: LongTensor of agent indices, shape [B]
            - inputs['theta']: AV heading angle, shape [B]
            - inputs['origin']: global origin position, shape [B, 2]

    Returns:
        global_hist: torch.Tensor of shape [B, 20, 2], absolute positions in global frame
    """
    x0 = x0.squeeze(1)  # [B, 20, 2]
    B, T, _ = x0.shape
    assert T == 20, "Expected 20 history frames"

    # Step 1: Recover AV-centric absolute pos starting from last frame (t=19)
    rel_hist = torch.zeros_like(x0)  # [B, 20, 2]
    rel_hist[:, -1, :] = 0  # Last frame displacement = 0 by definition

    # Reverse cumulative subtraction: x[t-1] = x[t] - disp[t]
    for t in reversed(range(T - 1)):
        rel_hist[:, t, :] = rel_hist[:, t + 1, :] - x0[:, t + 1, :]  # Displacement from t to t+1

    # Step 2: Get agent last observed position in AV-centric frame
    last_rot = inputs['positions'][inputs['agent_index'], 19]  # [B, 2]

    # Step 3: Add anchor point to get AV-centric absolute trajectory
    abs_rot = rel_hist + last_rot.unsqueeze(1)  # [B, 20, 2]


    if reverse_rotation:
        # Step 4: Construct inverse of AV-centric rotation matrix
        theta = inputs['theta']  # [B]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        scene_R = torch.stack([
            torch.stack([cos_t, -sin_t], dim=1),
            torch.stack([sin_t,  cos_t], dim=1)
        ], dim=1)  # [B, 2, 2]
        inv_scene_R = scene_R.transpose(1, 2)  # [B, 2, 2]

        # Step 5: Rotate AV-centric trajectory back to global frame
        global_rot = torch.bmm(abs_rot, inv_scene_R)  # [B, 20, 2]

        # Step 6: Add AV global origin to obtain full global trajectory
        origin = inputs['origin']  # [B, 2]
        global_hist = global_rot + origin.unsqueeze(1)  # [B, 20, 2]
        return global_hist
    

    return abs_rot