
import torch
import torch.nn as nn
#from diffusion_planner.model.module.encoder import Encoder
from diffusion_planner.model.module.decoder import Decoder
from diffusion_planner.utils.utils import reconstruct_absolute_position_from_last_frame
from diffusion_planner.random_mask import random_mask_agent_history, mask_x_gt_by_lopt, recalculate_masks
from diffusion_planner.model.module.gating import gating, LoptDecoder
from models import GlobalInteractor
from models import LocalEncoder
from models import customEncoder
from models import MLPDecoder
from diffusion_planner.utils.utils import TemporalData
from copy import deepcopy
from losses import LaplaceNLLLoss
from losses import SoftTargetCrossEntropyLoss
from models.MLP_recon import MLPReconstructor

import torch.nn.functional as F
import copy

class Traj_Diffusion(nn.Module):
    def __init__(self, config):
        super().__init__()

        

        self.encoder = HiVT_Encoder(config)
        self.decoder = Diffusion_Planner_Decoder(config)
        # MLP decoder for X_0 recon
        # self.decoder = MLPReconstructor(local_channels=config.embed_dim,global_channels=config.embed_dim)
        self.pred_decoder = MLPDecoder(local_channels=config.embed_dim,
                                  global_channels=config.embed_dim,
                                  future_steps=config.future_steps,
                                  num_modes=config.num_modes,
                                  uncertain=True)
        self.rotate = config.rotate
        self.random_mask = config.random_mask
        self.device = torch.device(config.device)
        self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')
        
        # separate encoder for training
        self.diffusion_encoder =  Customized_Encoder(config)
        
        # self.gating = gating(embed_dim=config.embed_dim, hidden_size=256)
        # gating module 
        self.gating = LoptDecoder(embed_dim=config.embed_dim)


    def set_stage(self, stage: str):
        assert stage in ("recon", "pred", "joint")
        self.stage = stage


    @property
    def sde(self):
        return self.decoder.decoder.sde
    
    # expect HiVt preprocessed data as input
    def forward(self, inputs: TemporalData):
        if self.random_mask:
            inputs = random_mask_agent_history(inputs, min_keep=2, history_steps=20)

        
        
        

        ################### rotate preprocess#######################################
        if self.rotate and 'rotate_mat' not in inputs:
            rotate_mat = torch.empty(inputs.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(inputs['rotate_angles'])
            cos_vals = torch.cos(inputs['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            if inputs.y is not None:
                inputs.y = torch.bmm(inputs.y, rotate_mat)
            inputs['rotate_mat'] = rotate_mat
        else:
            inputs['rotate_mat'] = None
        #############################################################################

        # NOTE add random mask
        
        
        if self.stage == "recon":
            local_embedding , cls_token  = self.diffusion_encoder(inputs) # [N, 20, D], [N, D]
            
            
            L_opt, mask = self.gating(cls_token) # [B], [B, T_max]
            # import pdb; pdb.set_trace()

            decoder_outputs = self.decoder(local_embedding, inputs)
            
            x0 = decoder_outputs["x0"].squeeze(1)  # [B, T, 2]
            return x0, decoder_outputs
        
        # Prediction‐only stage: skip diffusion decoder,
        # feed *ground‐truth* inputs.x into prediction head
        if self.stage == "pred":
            # re-encode original history
            _, local_embed, global_embed = self.encoder(inputs)
            y_hat, pi = self.pred_decoder(local_embed, global_embed)
            return y_hat, pi
        
        encoder_outputs, cls_token = self.diffusion_encoder(inputs)
        
        # gating module to get L_opt and mask for agent
        L_opt, mask = self.gating(encoder_outputs) # [B], [B, T_max]
        L_opt = L_opt[inputs['agent_index']]  # [B] # get the L_opt for the agent
        mask = mask[inputs['agent_index']]  # [B, T_max] # get the mask for the agent
        
        inputs['L_opt'] = L_opt
        inputs['mask'] = mask

        # reconstruction module
        decoder_outputs = self.decoder(encoder_outputs, inputs, L_opt)
        x0 = decoder_outputs['x0']  # [B, T, 2]
        
        x0 = x0.squeeze(1)  # [B, T, 2]


        
        # get gt for recon loss
        gt = decoder_outputs['gt']  # [B, T, 2]

        
        # masked_x_gt = mask_x_gt_by_lopt( gt, L_opt)  # [B, T, 2]
        
       
        mask= mask.unsqueeze(-1)  # [B, 20, 1]
        


        decoder_outputs['gt'] = torch.where(mask, torch.zeros_like(decoder_outputs['gt']), decoder_outputs['gt'])  # mask gt in decoder outputs
        inputs['x_gt'] = decoder_outputs['gt'] 
        decoder_outputs['score'] = torch.where(mask, torch.zeros_like(decoder_outputs['score']), decoder_outputs['score'])  # [B, T, 2] # mask the score by L_opt
        
        # NOTE: debug here , fix the gt with Lopt mask 
        # import pdb; pdb.set_trace()
        x0 = torch.where(mask, torch.zeros_like(x0), x0)  # [B, T, 2] # mask the x0 by L_opt
        # recalculate the padding mask and bos mask for reconstructed history
        # L_opt = L_opt.floor().clamp(min=2, max=20).long()  # [B]
        inputs = recalculate_masks(inputs, L_opt, history_steps=20) # [B, T_max]

        # 3) Shallow‐copy inputs so we don’t overwrite the original
        inputs2 = copy.copy(inputs)

        # 4) Clone x to preserve grad into x0
        x_clone = inputs.x.clone()
        # 5) Replace only agent's slice with x0
        
        x_clone[inputs.agent_index] = x0
        inputs2.x = x_clone

        
        
        #######################################################################################################
        # NOTE : recaluate the position of the agent, and keep heading 
        # preserve position[18] [19
        recal_his_pos = reconstruct_absolute_position_from_last_frame(x0, inputs2) #[B,20,2]
        inputs2['positions'][inputs2['agent_index'], :20] = recal_his_pos
        ##############################################################################################

        
        
        # 6) Re‐encode with reconstructed history for prediction
        _, local_embed, global_embed = self.encoder(inputs2)
        y_hat, pi = self.pred_decoder(local_embed=local_embed, global_embed=global_embed)
        
        # inputs.x[inputs.agent_index] = x0

        # _ , local_embed, global_embed = self.encoder(inputs)
        # # prediction head
        # y_hat, pi = self.pred_decoder(local_embed=local_embed, global_embed=global_embed)


        return encoder_outputs, decoder_outputs,y_hat, pi, inputs2
    
    def compute_loss(self, y_hat, pi, inputs: TemporalData):
        reg_mask = ~inputs['padding_mask'][:, 20:]
        valid_steps = reg_mask.sum(dim=-1)
        cls_mask = valid_steps > 0
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - inputs.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(inputs.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], inputs.y[reg_mask])
        soft_target = F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()
        cls_loss = self.cls_loss(pi[cls_mask], soft_target)
        loss = reg_loss + cls_loss
        return loss, reg_loss, cls_loss
    
    def validation_step(self, data, batch_idx):
        _,_, y_hat, pi = self(data)
        reg_mask = ~data['padding_mask'][:, 20:]
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        self.log('val_reg_loss', reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        y_hat_agent = y_hat[:, data['agent_index'], :, : 2]
        y_agent = data.y[data['agent_index']]
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        best_mode_agent = fde_agent.argmin(dim=0)
        y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(data.num_graphs)]
        self.minADE.update(y_hat_best_agent, y_agent)
        self.minFDE.update(y_hat_best_agent, y_agent)
        self.minMR.update(y_hat_best_agent, y_agent)
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minMR', self.minMR, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))


    def configure_optimizers(self,config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": config.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=config.lr, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

# use backbone encoder to extract features from the input data
class HiVT_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = torch.device(config.device)

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
        # NOTE avoid rotate again if already rotated
        
        local_embed = self.local_encoder(data=data)
        global_embed = self.global_interactor(data=data, local_embed=local_embed)
        encoder_outputs = torch.cat((local_embed,global_embed), dim=-1) # [N,2*embed_dim]
        return encoder_outputs , local_embed, global_embed
    

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

    def forward(self, encoder_outputs, inputs, L_opt):
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
    
        decoder_outputs = self.decoder(encoder_outputs, inputs,L_opt)
        
        return decoder_outputs


class Customized_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = torch.device(config.device)

        self.historical_steps = config.historical_steps
        self.future_steps = config.future_steps
        self.num_modes = config.num_modes
        self.rotate = config.rotate
        # self.parallel = config.parallel
        # self.lr = config.lr
        # self.weight_decay = config.weight_decay
        # self.T_max = config.T_max
        self.local_encoder = customEncoder(historical_steps=config.historical_steps,
                                          node_dim=config.node_dim,
                                          edge_dim=config.edge_dim,
                                          embed_dim=config.embed_dim,
                                          num_heads=config.num_heads,
                                          dropout=config.dropout,
                                          num_temporal_layers=config.num_temporal_layers,
                                          local_radius=config.local_radius,
                                          parallel=config.parallel)
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
        # NOTE avoid rotate again if already rotated
        
        local_embed = self.local_encoder(data=data)
        # print("local_embed shape: ", local_embed.shape)
        return local_embed

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
