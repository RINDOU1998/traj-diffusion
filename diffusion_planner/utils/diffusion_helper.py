import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Any
from collections import Counter

def sample_av_history(batch: Data, T: int = 20) -> Tensor:
    """
    Extract AV historical trajectories from batched TemporalData.

    Args:
        batch (Data): Batched PyG TemporalData object.
        T (int): Number of historical time steps.

    Returns:
        Tensor: AV trajectories of shape [B, 1, T, 2]
    """
    x = batch.x                    # [B*N, T, 2]
    batch_vec = batch.batch        # [B*N]
    av_indices = batch.av_index    # [B]

    B = batch_vec.max().item() + 1
    print("B :", B)
    print("x.shape :", x.shape)

    av_trajs = x[av_indices]       # [B, T, 2]
    av_trajs = av_trajs.unsqueeze(1)  # [B, 1, T, 2]
    #######################################################
  
    counts = Counter(batch_vec.tolist())  # Counts how many times each graph ID appears
    num_nodes_list = [counts[i] for i in range(max(counts.keys()) + 1)]
    print("Number of nodes per graph:", num_nodes_list)
    
    # av_trajs = []

    # for g_id in range(B):
    #     node_indices = (batch_vec == g_id).nonzero(as_tuple=True)[0]
    #     av_local_index = av_indices[g_id]
    #     av_global_index = node_indices[av_local_index]
    #     av_trajs.append(x[av_global_index])  # [T, 2]

    # av_trajs = torch.stack(av_trajs, dim=0).unsqueeze(1)  # [B, 1, T, 2]
    return av_trajs


def extract_av_indices(batch: Data) -> Tensor:
    """
    Extract global indices of AV nodes from batched TemporalData.

    Args:
        batch (Data): Batched PyG TemporalData object.

    Returns:
        Tensor: AV global indices of shape [B]
    """
    batch_vec = batch.batch
    av_indices = batch.av_index
    B = batch_vec.max().item() + 1

    av_global_indices = []
    for g_id in range(B):
        node_indices = (batch_vec == g_id).nonzero(as_tuple=True)[0]
        av_local_index = av_indices[g_id]
        av_global_index = node_indices[av_local_index]
        av_global_indices.append(av_global_index)

    return torch.stack(av_global_indices, dim=0)  # [B]


def extract_av_embeddings(encoder_outputs: Tensor, batch: Data) -> Tensor:
    """
    Extract AV embeddings from encoder output using AV global indices.

    Args:
        encoder_outputs (Tensor): Shape [N, D] — encoder output for all agents.
        batch (Data): Batched PyG TemporalData object.

    Returns:
        Tensor: AV embeddings of shape [B, D]
    """
    av_global_indices = extract_av_indices(batch)  # [B]
    return encoder_outputs[av_global_indices]      # [B, D]


def extract_agent_type(batch_vec: Tensor, av_indices: Tensor, num_graphs: int) -> Tensor:
    """
    Create an agent type tensor (0 for AV, 1 for other agents).

    Args:
        batch_vec (Tensor): [B*N] — Graph ID for each node.
        av_indices (Tensor): Global AV indices of shape [B].
        num_graphs (int): Number of graphs in the batch.

    Returns:
        Tensor: Agent type tensor [B*N], where 0 = AV, 1 = other agents
    """
    agent_type = torch.ones_like(batch_vec)  # Default: 1 (non-AV)
    agent_type[av_indices] = 0  # Set AV nodes to 0
    return agent_type
