import torch
import torch.nn as nn
from torch_geometric.data import Data

def sample_av_history(batch, T=20):
    """

    Extracts AV historical trajectory from batched TemporalData.
    Returns shape [B, 1, T, 2]
    """
    x = batch.x                    # [B*N, T, 2]
    batch_vec = batch.batch        # [B*N]
    av_indices = batch.av_index    # [B] or list of AV indices per graph

    B = batch_vec.max().item() + 1
    av_trajs = []
    for g_id in range(B):
        node_indices = (batch_vec == g_id).nonzero(as_tuple=True)[0]
        av_local_index = av_indices[g_id]
        av_global_index = node_indices[av_local_index]
        av_trajs.append(x[av_global_index])  # [T, 2]

    av_trajs = torch.stack(av_trajs, dim=0).unsqueeze(1)  # [B, 1, T, 2]
    return av_trajs

def extract_av_indices(batch):
    """
    Args:
        batch: a batched TemporalData object with `batch.batch` and `batch.av_index`

    Returns:
        av_global_indices: Tensor of shape [B], containing the global indices of AV nodes
    """
    batch_vec = batch.batch           # [N] → maps each node to its graph
    B = batch_vec.max().item() + 1    # batch size
    av_global_indices = []

    for g_id in range(B):
        # Get global indices of nodes in graph g_id
        node_indices = (batch_vec == g_id).nonzero(as_tuple=True)[0]  # [num_nodes_in_graph]
        av_local_index = batch.av_index[g_id]  # local AV index in this graph
        av_global_index = node_indices[av_local_index]  # global index in encoder_outputs
        av_global_indices.append(av_global_index)

    av_global_indices = torch.stack(av_global_indices, dim=0)  # [B]
    return av_global_indices


def extract_av_embeddings(encoder_outputs, batch):
    """
    Args:
        encoder_outputs: [ N, 2*D]  (output of HiVT encoder)
        batch: a batched TemporalData object with `batch.batch` and `batch.av_index`

    Returns:
        av_embeddings: Tensor of shape [B, 2*D]
    """
    av_global_indices = extract_av_indices(batch)  # [B]
    
    # Index encoder_outputs: [B, 2*D]
    av_embeddings = encoder_outputs[av_global_indices, :]  # [ B, 2*D]

    return av_embeddings


