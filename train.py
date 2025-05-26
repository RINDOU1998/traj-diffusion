import os
import torch
import argparse
from torch import optim
from timm.utils import ModelEma
from torch.utils.data import  DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader

from diffusion_planner.model.diffusion_planner import Diffusion_Planner

from diffusion_planner.utils.train_utils import set_seed, save_model, resume_model
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from diffusion_planner.utils.tb_log import TensorBoardLogger as Logger
#from diffusion_planner.utils.data_augmentation import StatePerturbation
#from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils import ddp
# from diffusion_planner.train_epoch import train_epoch 
from diffusion_planner.val_epoch import validation_epoch
from diffusion_planner.rec_val_epoch import validation_epoch as rec_validation_epoch
from diffusion_planner.train_epoch_2stage import train_epoch
#from datamodules import ArgoverseV1DataModule
from datasets import ArgoverseV1Dataset
from traj_diffusion import Traj_Diffusion



def boolean(v): 
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def add_model_specific_args(parent_parser):
    parser = parent_parser.add_argument_group('HiVT')
    parser.add_argument('--historical_steps', type=int, default=20)
    parser.add_argument('--future_steps', type=int, default=30)
    parser.add_argument('--num_modes', type=int, default=6)
    parser.add_argument('--rotate', type=bool, default=True)
    parser.add_argument('--node_dim', type=int, default=2)
    parser.add_argument('--edge_dim', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, required=True)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_temporal_layers', type=int, default=4)
    parser.add_argument('--num_global_layers', type=int, default=3)
    parser.add_argument('--local_radius', type=float, default=50)
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--T_max', type=int, default=64)
    return parent_parser


def get_args():
    parser = argparse.ArgumentParser(description='Training')
    # HiVT args
    parser.add_argument('--do_validation', type=bool, default=False , help='Run validation only instead of training')
    

    parser.add_argument('--root', type=str, required=True)
    #parser.add_argument('--val_root', type=str, required=True)
    # parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=True)
    #parser.add_argument('--num_workers', type=int, default=8)
    # parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    #parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=5)
    parser = add_model_specific_args(parser)


    # Arguments
    
    parser.add_argument('--name', type=str, help='log name (default: "diffusion-planner-training")', default="diffusion-planner-training")
    parser.add_argument('--save_dir', type=str, help='save dir for model ckpt', default=".")

    # Data

    # parser.add_argument('--train_set', type=str, help='path to train data', default=None)
    # parser.add_argument('--train_set_list', type=str, help='data list of train data', default=None)

    # parser.add_argument('--future_len', type=int, help='number of time point', default=80)
    # parser.add_argument('--time_len', type=int, help='number of time point', default=21)

    # parser.add_argument('--agent_state_dim', type=int, help='past state dim for agents', default=11)
    # parser.add_argument('--agent_num', type=int, help='number of agents', default=32)

    # parser.add_argument('--static_objects_state_dim', type=int, help='state dim for static objects', default=10)
    # parser.add_argument('--static_objects_num', type=int, help='number of static objects', default=5)

    # parser.add_argument('--lane_len', type=int, help='number of lane point', default=20)
    # parser.add_argument('--lane_state_dim', type=int, help='state dim for lane point', default=12)
    # parser.add_argument('--lane_num', type=int, help='number of lanes', default=70)

    # parser.add_argument('--route_len', type=int, help='number of route lane point', default=20)
    # parser.add_argument('--route_state_dim', type=int, help='state dim for route lane point', default=12)
    # parser.add_argument('--route_num', type=int, help='number of route lanes', default=25)


    # DataLoader parameters
    parser.add_argument('--augment_prob', type=float, help='augmentation probability', default=0.5)
    parser.add_argument('--normalization_file_path', default='normalization.json', help='filepath of normalizaiton.json', type=str)
    parser.add_argument('--use_data_augment', default=False, type=boolean)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)
    
    # Training
    parser.add_argument('--recon_epochs', type=int, default=64,
                    help="# epochs to train reconstruction only")
    parser.add_argument('--pred_epochs',  type=int, default=64,
                    help="# epochs to train prediction only")
    parser.add_argument('--seed', type=int, help='fix random seed', default=3407)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=30)
    parser.add_argument('--save_utd', type=int, help='save frequency', default=20)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 32)', default=32)
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 5e-4)', default=5e-4)

    parser.add_argument('--warm_up_epoch', type=int, help='number of warm up', default=1)
    parser.add_argument('--encoder_drop_path_rate', type=float, help='encoder drop out rate', default=0.1)
    parser.add_argument('--decoder_drop_path_rate', type=float, help='decoder drop out rate', default=0.1)

    parser.add_argument('--alpha_recon', type=float, help='coefficient of diffusion reconstruction (default: 1.0)', default=1)

    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')

    parser.add_argument('--use_ema', default=False, type=boolean)

    # Model
    parser.add_argument('--encoder_depth', type=int, help='number of encoding layers', default=3)
    parser.add_argument('--decoder_depth', type=int, help='number of decoding layers', default=3)
    #parser.add_argument('--num_heads', type=int, help='number of multi-head', default=6)
    parser.add_argument('--hidden_dim', type=int, help='hidden dimension', default=256)
    parser.add_argument('--diffusion_model_type', type=str, help='type of diffusion model [x_start, score]', choices=['score', 'x_start'], default='x_start')

    # decoder
    parser.add_argument('--predicted_neighbor_num', type=int, help='number of neighbor agents to predict', default=10)
    parser.add_argument('--resume_model_path', type=str, help='path to resume model', default=None)

    parser.add_argument('--use_wandb', default=False, type=boolean)
    parser.add_argument('--notes', default='', type=str)

    # distributed training parameters
    parser.add_argument('--ddp', default=False, type=boolean, help='use ddp or not')
    parser.add_argument('--port', default='22323', type=str, help='port')

    args = parser.parse_args()

    #args.state_normalizer = StateNormalizer.from_json(args)
    args.state_normalizer = None
    #args.observation_normalizer = ObservationNormalizer.from_json(args)
    args.observation_normalizer = None
    return args

def model_training(args):

    # init ddp
    #########################DDP set up distributed training#########################
    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)

    if global_rank == 0:
        # Logging
        print("------------- {} -------------".format(args.name))
        print("Batch size: {}".format(args.batch_size))
        print("Learning rate: {}".format(args.learning_rate))
        print("Use device: {}".format(args.device))

        if args.resume_model_path is not None:
            save_path = args.resume_model_path
        else:
            from datetime import datetime
            time = datetime.now()
            time = time.strftime("%Y-%m-%d-%H:%M:%S")

            save_path = f"{args.save_dir}/training_log/{args.name}/{time}/"
            os.makedirs(save_path, exist_ok=True)

        # Save args
        args_dict = vars(args)
        args_dict = {k: v if not isinstance(v, (StateNormalizer, ObservationNormalizer)) else v.to_dict() for k, v in args_dict.items() }

        from mmengine.fileio import dump
        dump(args_dict, os.path.join(save_path, 'args.json'), file_format='json', indent=4)
    else:
        save_path = None

    # set seed
    set_seed(args.seed + global_rank)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size
    
    # set up data loaders
    ################## modify the datasource as the desired dataset############
    # no train_transform and aug 


    # aug = StatePerturbation(augment_prob=args.augment_prob, device=args.device) if args.use_data_augment else None
    # train_set = DiffusionPlannerData(args.train_set, args.train_set_list, args.agent_num, args.predicted_neighbor_num, args.future_len)
    # train_sampler = DistributedSampler(train_set, num_replicas=ddp.get_world_size(), rank=global_rank, shuffle=True)
    # train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=batch_size//ddp.get_world_size(), num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    
    # create train set , train_sampler, and train_loader from  ArgoverseV1Dataset  and ArgoverseV1DataModule
    # aug = StatePerturbation(augment_prob=args.augment_prob, device=args.device) if args.use_data_augment else None
    aug = None
    train_set = ArgoverseV1Dataset(args.root, 'train', None, args.local_radius )

    val_dataset = ArgoverseV1Dataset(args.root, 'val', None, args.local_radius )
    # train_sampler = DistributedSampler(train_set, num_replicas=ddp.get_world_size(), rank=global_rank, shuffle=True)
    # sampler=train_sampler,
    train_loader = DataLoader(train_set,  batch_size=batch_size//ddp.get_world_size(), num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    val_loader = DataLoader(val_dataset,  batch_size=batch_size//ddp.get_world_size(),shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)


    if global_rank == 0:
        print("Dataset Prepared: {} train data\n".format(len(train_set)))

    if args.ddp:
        torch.distributed.barrier()

    # set up model
    #diffusion_planner = Diffusion_Planner(args)
    #diffusion_planner = diffusion_planner.to(rank if args.device == 'cuda' else args.device)

    #set up model Traj-dffusion and ddp 
    diffusion_planner = Traj_Diffusion(args)
    diffusion_planner = diffusion_planner.to(rank if args.device == 'cuda' else args.device)



    if args.ddp:
        diffusion_planner = DDP(diffusion_planner, device_ids=[rank])

    if args.use_ema:
        model_ema = ModelEma(
            diffusion_planner,
            decay=0.999,
            device=args.device,
        )
    
    if global_rank == 0:
        print("Model Params: {}".format(sum(p.numel() for p in ddp.get_model(diffusion_planner, args.ddp).parameters())))

    # optimizer
    params = [{'params': ddp.get_model(diffusion_planner, args.ddp).parameters(), 'lr': args.learning_rate}]

    optimizer = optim.AdamW(params)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, train_epochs, args.warm_up_epoch)

    if args.resume_model_path is not None:
        print(f"Model loaded from {args.resume_model_path}")
        diffusion_planner, optimizer, scheduler, init_epoch, wandb_id, model_ema = resume_model(args.resume_model_path, diffusion_planner, optimizer, scheduler, model_ema, args.device)
    else:
        init_epoch = 0
        wandb_id = None

    # logger
    wandb_logger = Logger(args.name, args.notes, args, wandb_resume_id=wandb_id, save_path=save_path, rank=global_rank) 

    if args.ddp:
        torch.distributed.barrier()


    best_k_models = []
    K = args.save_top_k

    recon_losses = []
    pred_losses = []
    joint_losses = []

    # Phase 1: reconstruction
    if args.recon_epochs > 0:
        diffusion_planner.set_stage("recon")
        for e in range(args.recon_epochs):
            print(f"[Recon] Epoch {e+1}/{args.recon_epochs}")
            train_loss, train_total_loss = train_epoch(train_loader, diffusion_planner, optimizer, args)
            wandb_logger.log_metrics({f"train_loss/{k}": v for k, v in train_loss.items()}, step=e+1)
            recon_losses.append(train_total_loss)
            ade  = rec_validation_epoch(diffusion_planner, val_loader, args.device)
            wandb_logger.log_metrics({"val/ade": ade}, step=e+1)
            


    # Phase 2: prediction head
    if args.pred_epochs > 0:
        diffusion_planner.set_stage("pred")
        for e in range(args.pred_epochs):
            print(f"[Pred] Epoch {e+1}/{args.pred_epochs}")
            train_loss, train_total_loss = train_epoch(train_loader, diffusion_planner, optimizer, args)
            pred_losses.append(train_total_loss)

    # Phase 3: joint finetuning
    diffusion_planner.set_stage("joint")
    # begin training
    for epoch in range(init_epoch, train_epochs):

        # warm_up learning rate 

        # if(epoch>args.warm_up_epoch):
        #     args.alpha_recon = 0.01

        if global_rank == 0:
            print(f"Epoch {epoch+1}/{train_epochs}")
        train_loss, train_total_loss = train_epoch(train_loader, diffusion_planner, optimizer, args,  aug)
        joint_losses.append(train_total_loss)
        print("Train  Loss: ", train_loss)
        
        if global_rank == 0:
            #logging training
            lr_dict = {'lr': optimizer.param_groups[0]['lr']}
            wandb_logger.log_metrics({f"train_loss/{k}": v for k, v in train_loss.items()}, step=epoch+1)
            wandb_logger.log_metrics({f"lr/{k}": v for k, v in lr_dict.items()}, step=epoch+1)
            print({f"train_loss/{k}": v for k, v in train_loss.items()} , "epoch: ", epoch+1)
            print({f"lr/{k}": v for k, v in lr_dict.items()}, "epoch: ", epoch+1)
            # validation
            val_ade, val_fde, val_mr = validation_epoch(diffusion_planner, val_loader, args.device)
            wandb_logger.log_metrics({"val/ade": val_ade, "val/fde": val_fde, "val/mr": val_mr}, step=epoch+1)

            if (epoch+1) % args.save_utd == 0:
                # save model at the end of epoch
                train_losses = {
                    'joint_losses': joint_losses,
                    'recon_losses': recon_losses,
                    'pred_losses': pred_losses
                }
                save_model(diffusion_planner, optimizer, scheduler, save_path, epoch, train_losses, wandb_logger.id)
                print(f"Model saved in {save_path}\n")
            else:
                # Save top-k model by ADE
                best_k_models.append((val_ade, epoch))
                best_k_models = sorted(best_k_models, key=lambda x: x[0])[:K]

                for _, top_epoch in best_k_models:
                    if epoch == top_epoch:
                        save_model(diffusion_planner, optimizer, scheduler, save_path, epoch, train_total_loss, wandb_logger.id)
                        print(f"Top-{K} Model saved at epoch {epoch+1} with ADE {val_ade:.4f}")
        scheduler.step()
        #train_sampler.set_epoch(epoch + 1)
    
    print("Training finished")


def model_validation(args):
    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)

    if global_rank == 0:
        print(f"\n🧪 Running Validation for {args.name} on {args.device}")

    val_set = ArgoverseV1Dataset(args.root, 'val', None, args.local_radius)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    
   

    diffusion_planner = Traj_Diffusion(args).to(args.device)

    if args.resume_model_path:
        print(f"🔄 Loading checkpoint from: {args.resume_model_path}")
        diffusion_planner, _, _, _, _, _ = resume_model(
            args.resume_model_path, diffusion_planner, None, None, None, args.device
        )
    
    # run validation
    val_ade, val_fde, val_mr, rec_loss = validation_epoch(diffusion_planner, val_loader, args.device)

    if global_rank == 0:
        print(f"\n✅ Validation Metrics:\n - ADE: {val_ade:.4f}\n - FDE: {val_fde:.4f}\n - Miss Rate: {val_mr:.4f} - Reconstruction Loss: {rec_loss:.4f}")




if __name__ == "__main__":
    args = get_args()

    if args.do_validation:
        model_validation(args)
    else:
        model_training(args)
