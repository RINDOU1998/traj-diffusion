
import torch
import torch.nn as nn

#from diffusion_planner.model.module.encoder import Encoder
from diffusion_planner.model.module.decoder import Decoder
from models import GlobalInteractor
from models import LocalEncoder
from diffusion_planner.utils.utils import TemporalData

class Traj_Diffusion(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = HiVT_Encoder(config)
        self.decoder = Diffusion_Planner_Decoder(config)

    @property
    def sde(self):
        return self.decoder.decoder.sde
    
    # expect HiVt preprocessed data as input
    def forward(self, inputs: TemporalData):

        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(encoder_outputs, inputs)

        return encoder_outputs, decoder_outputs


# use backbone encoder to extract features from the input data
class HiVT_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.historical_steps = config.historical_steps
        self.future_steps = config.future_steps
        self.num_modes = config.num_modes
        self.rotate = config.rotate
        # self.parallel = config.parallel
        # self.lr = config.lr
        # self.weight_decay = config.weight_decay
        # self.T_max = config.T_max
        self.local_encoder = LocalEncoder(historical_steps=config.historical_steps,
                                          node_dim=config.node_dim,
                                          edge_dim=config.edge_dim,
                                          embed_dim=config.embed_dim,
                                          num_heads=config.num_heads,
                                          dropout=config.dropout,
                                          num_temporal_layers=config.num_temporal_layers,
                                          local_radius=config.local_radius,
                                          parallel=config.parallel)
        self.global_interactor = GlobalInteractor(historical_steps=config.historical_steps,
                                                  embed_dim=config.embed_dim,
                                                  edge_dim=config.edge_dim,
                                                  num_modes=config.num_modes,
                                                  num_heads=config.num_heads,
                                                  num_layers=config.num_global_layers,
                                                  dropout=config.dropout,
                                                  rotate=config.rotate)
        self.initialize_weights()

    #initialize weights of the model
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_basic_init)

    def forward(self, data: TemporalData):
        if self.rotate:
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(data['rotate_angles'])
            cos_vals = torch.cos(data['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            if data.y is not None:
                data.y = torch.bmm(data.y, rotate_mat)
            data['rotate_mat'] = rotate_mat
        else:
            data['rotate_mat'] = None

        local_embed = self.local_encoder(data=data)
        global_embed = self.global_interactor(data=data, local_embed=local_embed)
        encoder_outputs = torch.cat((local_embed,global_embed), dim=-1) # [N,2*embed_dim]
        return encoder_outputs
    

class Diffusion_Planner_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.decoder = Decoder(config)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.decoder.dit.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].bias, 0)

    def forward(self, encoder_outputs, inputs):
        ## TODO convert TemporalData into valid inputs for diffusion decoder
        #inputs: Dict
        '''
        {
        ...
        "ego_current_state": current ego states,            
        "neighbor_agent_past": past and current neighbor states,  

        [training-only] "sampled_trajectories": sampled current-future ego & neighbor states,        [B, P, 1 + V_future, 4]
        [training-only] "diffusion_time": timestep of diffusion process $t \in [0, 1]$,              [B]
        ...
        }
        '''
    
        decoder_outputs = self.decoder(encoder_outputs, inputs)
        
        return decoder_outputs
    

# HiVT hyperparameters
# @staticmethod
# def add_model_specific_args(parent_parser):
#     parser = parent_parser.add_argument_group('HiVT')
#     parser.add_argument('--historical_steps', type=int, default=20)
#     parser.add_argument('--future_steps', type=int, default=30)
#     parser.add_argument('--num_modes', type=int, default=6)
#     parser.add_argument('--rotate', type=bool, default=True)
#     parser.add_argument('--node_dim', type=int, default=2)
#     parser.add_argument('--edge_dim', type=int, default=2)
#     parser.add_argument('--embed_dim', type=int, required=True)
#     parser.add_argument('--num_heads', type=int, default=8)
#     parser.add_argument('--dropout', type=float, default=0.1)
#     parser.add_argument('--num_temporal_layers', type=int, default=4)
#     parser.add_argument('--num_global_layers', type=int, default=3)
#     parser.add_argument('--local_radius', type=float, default=50)
#     parser.add_argument('--parallel', type=bool, default=False)
#     parser.add_argument('--lr', type=float, default=5e-4)
#     parser.add_argument('--weight_decay', type=float, default=1e-4)
#     parser.add_argument('--T_max', type=int, default=64)
#     return parent_parser
