import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

class MLPReconstructor(nn.Module):
    def __init__(self,
                 local_channels: int,
                 global_channels: int,
                 history_steps: int = 20):
        super(MLPReconstructor, self).__init__()
        self.input_dim = local_channels 
        self.history_steps = history_steps

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.LayerNorm(self.input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim, self.input_dim),
            nn.LayerNorm(self.input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim,  2)  # [B, 20*2]
        )

        self.apply(init_weights)

    def forward(self, encoder_outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            local_embed:  [B, D]
            global_embed: [B, D]

        Returns:
            x0: [B, 20, 2] — Reconstructed displacement history

        """

        # agent_embed = encoder_outputs[inputs['agent_index'], :] # [B,D]
        # x0 = self.mlp(agent_embed)                                    # [B, 40]
        # x0 = x0.view(-1, self.history_steps, 2)                 # [B, 20, 2]
        # x0.unsqueeze_(1)  # [B, 1, 20, 2]
        # gt = inputs['x'][inputs['agent_index'], :,:2].unsqueeze(1)  # [B, 1, 20, 2]
        # # keep the last two frame pos and last displacement
        # x0[:, :, -1, :] = gt[:, :, -1, :] # [B, P, 20, 2]
        
        agent_embed = encoder_outputs[inputs['agent_index'], :] # [B,T,D ]
        x0 = self.mlp(agent_embed)                                    # [B, T,2]
        # x0 = x0.view(-1, self.history_steps, 2)                 # [B, 20, 2]
        x0.unsqueeze_(1)  # [B, 1, 20, 2]
        gt = inputs['x'][inputs['agent_index'], :,:2].unsqueeze(1)  # [B, 1, 20, 2]
        # keep the last two frame pos and last displacement
        x0[:, :, -1, :] = gt[:, :, -1, :] # [B, P, 20, 2]

        return  {
                    "score": x0,
                   
                    "std" : None,
                    "z" : None,
                    "gt" : gt,
                    "x0": x0
                }
