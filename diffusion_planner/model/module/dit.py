import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp
from torch_geometric.utils import to_dense_batch

def modulate(x, shift, scale, only_first=False):
    # if only_first:
    #     x_first, x_rest = x[:, :1], x[:, 1:]
    #     x = torch.cat([x_first * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), x_rest], dim=1)
    # else:
    #     x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    # fix the dimension of shift and scale
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x_first = x_first * (1 + scale[:, :1]) + shift[:, :1]
        x = torch.cat([x_first, x_rest], dim=1)
    else:
        x = x * (1 + scale) + shift

    return x


def scale(x, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1))

    return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning for ego and Cross-Attention.
    """
    def __init__(self, dim=256, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)

        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.attn_mask = self.generate_square_subsequent_mask(20)

    def forward(self, x, cross_c, t_embed, batch_vec):
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cross_c).chunk(6, dim=2) # [B,T,D] token-wise modulation
        # adaln self attn
        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        


        # NOTE do not use the causal mask current

        # also fix th

        x = x + gate_msa * self.attn(modulated_x, modulated_x, modulated_x)[0]
        
        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp1(modulated_x)
        



        # cross attn with cross  
       
        #  cross c in shape [B, 20 ,D]
        #  x in shape [B, 1, D ]


        # context, mask = to_dense_batch(cross_c, batch_vec)  # [B, N_max, D], [B, N_max]

        #print("context shape:", context.shape)
        #print(" x shape:", x.shape)

        # x, _ = self.cross_attn(
        #     query=self.norm1(x),                # [T, B, D]
        #     key=cross_c,            # [T, B, D]
        #     value=cross_c           # [T, B, D]
        #     # key_padding_mask=~mask      # [B, N_max]
        # )
        #x = self.cross_attn(self.norm3(x), cross_c, cross_c)[0]
       
        #x = self.mlp2(self.norm4(x))  
        return x 

    @staticmethod
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask 
    
    
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x):
        B, P, _ = x.shape
        # NOTE remove  y as adaln guidance
        # shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        # x = modulate(self.norm_final(x), shift, scale)

        x = self.norm_final(x)
        x = self.proj(x)
        return x
    