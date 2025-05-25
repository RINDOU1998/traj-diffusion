from collections import OrderedDict
import copy
import math
import os
import pickle as pkl
from typing import Any, Dict, List, Optional, Tuple, Union
from argoverse.map_representation.map_api import ArgoverseMap
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, LinearRing
from shapely.affinity import affine_transform, rotate
import torch
import pandas as pd


def get_centerlines(am , raw_path: str, city, radius) -> List[List[np.ndarray]]:
    """
    Retrieve centerlines from a CSV file.

    Args:
        raw_path (str): Path to the CSV file containing centerline data.

    Returns:
        List[List[np.ndarray]]: Lane polylines for each agent.
    """
    centerlines = []
    df = pd.read_csv(raw_path)
    timestamps = list(np.sort(df['TIMESTAMP'].unique()))
    historical_timestamps = timestamps[: 20]
    historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]
    actor_ids = list(historical_df['TRACK_ID'].unique())
    df = df[df['TRACK_ID'].isin(actor_ids)]
    num_nodes = len(actor_ids)

    df_19 = df[df['TIMESTAMP'] == timestamps[19]]
    node_inds_19 = [actor_ids.index(actor_id) for actor_id in df_19['TRACK_ID']]
    node_positions_19 = torch.from_numpy(np.stack([df_19['X'].values, df_19['Y'].values], axis=-1)).float()
   
    lane_ids = set()
    for node_position in node_positions_19:
        lane_ids.update(am.get_lane_ids_in_xy_bbox(node_position[0], node_position[1], city, radius))

    for lane_id in lane_ids:
        lane_centerline = am.get_lane_segment_centerline(lane_id, city)[:, : 2]
        centerlines.append(lane_centerline)

    #print(centerlines.shape)
    return centerlines

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




def batch_output_to_np_list(inputs,inputs2, x0, y_hat, val_data_folder,radius=2.5):
    """
    Converts batched model outputs into lists of numpy arrays for visualization.
    
    Args:
        inputs (TemporalData): batched input
        x0 (Tensor): reconstructed AV history [B, 20, 2]
        y_hat (Tensor): predicted trajectories [F, N, 30, 4]
        val_data_folder (str): folder with .pkl centerline files

    Returns:
        Tuple:
            input_np_list: list of [N_i, 20, 2] numpy arrays (per graph)
            output_np_list: list of [N_i, F, 30, 2] numpy arrays (per graph)
            target_np_list: list of [N_i, 30, 2] numpy arrays (per graph)
            centerline_list: list of [list of np.ndarray] per graph
            city_name_list: list of city name strings per graph
    """
    input_np_list = []
    output_np_list = []
    target_np_list = []
    centerline_list = []
    city_name_list = []

    scene_list = []
    avm = ArgoverseMap()


    seq_ids = inputs.seq_id.cpu().numpy()
    for seq_id in seq_ids:

        raw_data_path = os.path.join(val_data_folder, str(seq_id) + '.csv')
        df = pd.read_csv(raw_data_path)

        timestamps = list(np.sort(df['TIMESTAMP'].unique()))
        historical_timestamps = timestamps[: 20]
        future_timestamps = timestamps[20:]
        # historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]
        # future_df = df[df['TIMESTAMP'].isin(future_timestamps)]

        agent_df = df[df['OBJECT_TYPE'] == 'AGENT']
        

        agent_his = agent_df[agent_df['TIMESTAMP'].isin(historical_timestamps)].sort_values('TIMESTAMP')
        agent_fut = agent_df[agent_df['TIMESTAMP'].isin(future_timestamps)].sort_values('TIMESTAMP')
        
        agent_his_pos = agent_his[['X', 'Y']].values  # shape [20, 2]
       
        
        agent_fut_pos = agent_fut[['X', 'Y']].values  # shape [30, 2]
        
        

        city = df['CITY_NAME'].values[0]

        lane_ids_set = set()
        for x, y in agent_his_pos:
            lane_ids = avm.get_lane_ids_in_xy_bbox(x, y, city, radius)
            lane_ids_set.update(lane_ids)

        for x, y in agent_fut_pos:
            lane_ids = avm.get_lane_ids_in_xy_bbox(x, y, city, radius)
            lane_ids_set.update(lane_ids)


        centerline_candidates = []
        for lane_id in lane_ids_set:
            centerline = avm.get_lane_segment_centerline(lane_id, city)[:, :2]
            
            centerline_candidates.append(centerline)
        


        
        scene_list.append( {"input": agent_his_pos.reshape(1, 20, 2),
                            "target": agent_fut_pos.reshape(1, 30, 2),
                            "centerlines": [centerline_candidates],
                            "city":[city]
                            } )

    
    
    # recalculate y_hat absolute pos
    # Step 0: y_hat only for agent
    y_hat = y_hat[:, inputs['agent_index'], :, :2]  # [F, B, 30, 2]
    inv_actor_rot = inputs2['rotate_mat'][inputs['agent_index'],:,:].transpose(1, 2)  # [B, 2, 2]
    # Step 1: Undo agent heading rotation
    # print("inv_actor_rot shape:", inv_actor_rot.shape)  # [B, 2, 2]
    F, B, H, _ = y_hat.shape 
    y_hat_flat = y_hat.reshape(F * B, H, 2)             # [F*N, H, 2]
    # print("y_hat_flat shape:", y_hat_flat.shape)  # [F*N, H, 2]
    inv_actor_rot = inv_actor_rot.unsqueeze(0).expand(F, -1, -1, -1)  # [F, B, 2, 2]
    # print("inv_actor_rot shape:", inv_actor_rot.shape)  # [F, B, 2, 2]
    inv_rot_flat = inv_actor_rot.reshape(F * B, 2, 2)     # [F*B, 2, 2]
    # print("inv_rot_flat shape:", inv_rot_flat.shape)  # [F*B, 2, 2]
    y_hat_scene = torch.bmm(y_hat_flat, inv_rot_flat)    # [F*N, H, 2]
    # print("y_hat_scene shape:", y_hat_scene.shape)  # [F*B, H, 2]
    # Step 2: Add last observed position
    x_last = inputs2['positions'][inputs['agent_index'], 19, :]  # [B, 2]
    # Repeat for F modes
    x_last_repeat = x_last.unsqueeze(0).expand(F, -1, -1)  # [F, B, 2]
    x_last_flat = x_last_repeat.reshape(F * B, 1, 2)       # [F*B, 1, 2]
    # Add to y_hat_scene
    y_hat_centered = y_hat_scene + x_last_flat             # [F*B, H, 2]
    # print("y_hat_centered shape:", y_hat_centered.shape)   # [F*B, H, 2]
    # Step 3: Undo AV-centric rotation    
    theta = inputs['theta']  # [B]
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    # Construct AV-centric rotation matrix (B, 2, 2)
    av_rot_mat = torch.stack([
        cos_theta, -sin_theta,
        sin_theta, cos_theta
    ], dim=1).view(B, 2, 2)
    # Inverse = transpose
    inv_scene_rot = av_rot_mat.transpose(1, 2)         # [B, 2, 2]
    # Repeat to match F modes
    inv_scene_rot = inv_scene_rot.unsqueeze(0).expand(F, -1, -1, -1)  # [F, B, 2, 2]
    inv_scene_rot_flat = inv_scene_rot.reshape(F * B, 2, 2)           # [F*B, 2, 2]
    # Apply inverse rotation
    y_hat_global = torch.bmm(y_hat_centered, inv_scene_rot_flat)     # [F*B, H, 2]
    # print( f"y_hat_global.shape : {y_hat_global.shape}")
    # Step 4: Add global origin
    origin = inputs['origin']  # [B, 2]
    origin_expanded = origin.unsqueeze(0).expand(F, -1, -1).reshape(F * B, 1, 2)  # [F * B, 1, 2]
    y_hat_global = y_hat_global + origin_expanded  # [F * B, H, 2]
    y_hat = y_hat_global.reshape(F,B,H,2)
    # print(f"y_hashapet:{y_hat.shape}")  
    #print(f"yhat after{y_hat}") 
    y_hat = y_hat.cpu().numpy()  # convert to NumPy if still in torch.Tensor

    
    # Result: list of length B; each item is a list of F arrays with shape [1, H, 2]
    y_hat_list = [
        [y_hat[f, b][:, :] for f in range(F)]  # [1, H, 2]
        for b in range(B)
    ]
    # print(len(y_hat_list))
    # print(len(y_hat_list[0]))
    

    #print(f"yhat:{y_hat_list[0][0].shape}")


    # #agent_index = inputs['agent_index'].cpu() 
    # #x0 = x0.squeeze(1)    # [B, 20, 2]
    # disp = inputs['x'][ inputs['agent_index'] ]  # [B, 20, 2]
    # print(f"x0:{x0}")
    # # step 1 cumulative  displacement
    # x0_absolute = disp.cumsum(dim=0)  
    # print(f"x0_absolute: {x0_absolute}")  # [B, 20, 2]
    # # step 2 undo agent heading rotation
    # theta = inputs['theta'] 
    # cos_theta = torch.cos(theta)
    # sin_theta = torch.sin(theta)
    # # Construct AV-centric rotation matrix (B, 2, 2)
    # av_rot_mat = torch.stack([
    #     cos_theta, -sin_theta,
    #     sin_theta, cos_theta
    # ], dim=1).view(B, 2, 2)
    # inv_rot = av_rot_mat.transpose(1, 2)    # [B, 2, 2]
    # x0_global = torch.bmm(x0_absolute, inv_rot)  # [B, 20, 2]
    # # step 3 add origin
    # x0_global = x0_global + origin.unsqueeze(1)
    # x0_global = x0_global.cpu().numpy()  # convert to NumPy if still in torch.Tensor
    # print(f"x0_global : {x0_global}")  # [B, 20, 2]

    # pos =  inputs['positions'][inputs['agent_index'], 19, :]


   
    # —— 1. 取出 agent 的位移历史 （已在 AV-centric 下，x[:,0]=0） ——
    # agent_idx = inputs['agent_index']
    # disp       = inputs['x'][agent_idx]            # [20, 2]

    x0 = x0.squeeze(1)    # [B, 20, 2]
    B, T, _ = x0.shape

    # —— 1. 累积位移变成 AV-centric 下的绝对轨迹（相对 agent 第一帧） ——
    rel_hist = x0.cumsum(dim=1)            # [B, 20, 2]

    # —— 2. 拿回 AV-centric 坐标下 agent 第0帧的位置 ——
    init_rot = inputs2['positions'][inputs['agent_index'], 0]  # [B, 2]

    # —— 3. 加回初始位置 ——
    abs_rot = rel_hist + init_rot.unsqueeze(1)        # [B, 20, 2]

    # —— 4. 构造 AV-centric 旋转矩阵并取逆 —— 
    theta = inputs['theta']                          # [B]
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    scene_R = torch.stack([
        torch.stack([cos_t, -sin_t], dim=1),
        torch.stack([sin_t,  cos_t], dim=1)
    ], dim=1)  # [B, 2, 2]

    inv_scene_R = scene_R.transpose(1, 2)             # [B, 2, 2]

    # —— 5. 旋转到原始全局坐标系 ——
    # 需要 reshape 后用 bmm
    # abs_rot_flat = abs_rot.reshape(B * T, 2)          # [B*T, 2]
    # inv_rot_flat = inv_scene_R.repeat_interleave(T, dim=0)  # [B*T, 2, 2]
    #global_rot = torch.bmm(abs_rot_flat.unsqueeze(1), inv_rot_flat).squeeze(1)  # [B*T, 2]
    #global_rot = global_rot.reshape(B, T, 2)          # [B, 20, 2]
    global_rot = torch.bmm(abs_rot, inv_scene_R)  # [B, 20, 2]
    # —— 6. 加回原始 AV 的全局位置（origin） ——
    origin = inputs['origin']                         # [B, 2]
    global_hist = global_rot + origin.unsqueeze(1)    # [B, 20, 2]
    global_hist = global_hist.cpu().numpy()


#####################################
    # disp = inputs['x'][ inputs['agent_index'] ]  # [B, 20, 2]
    #x0 = x0.unsqueeze(1) 
    #disp = x0 
    #print(f"disp.shape: {disp.shape}")  # [B, 20, 2]
    # disp = disp.unsqueeze(1)  # [B, 1, 20, 2]
    # result = reconstruct_absolute_position_from_last_frame(disp, inputs)
    # result = result.cpu().numpy()
    # print(f"recon absolute pos: {result}")
    # print(f"recon absolute pos shape: {result.shape}")  # [B, 20, 2]
    # abs_rt_gt = inputs["positions"][inputs["agent_index"], :20]  # [B, 2]
    # print(f"abs_rt_gt: {abs_rt_gt}")
    # print(f"abs_rt_gt shape: {abs_rt_gt.shape}")
    #agent_h = scene_list[0]["input"][0]
    #print("true absolu pos", agent_h)  # [20, 2]

    #dif = agent_h - result
    #print("dif:", dif)  # [20, 2]
############################################
    for i in range(B):
        scene_list[i]["output"] = [y_hat_list[i]]
        scene_list[i]["recon"] = global_hist[i]

    
    
    return scene_list



def viz_predictions(
    input_: np.ndarray,
    output: List[np.ndarray],
    target: np.ndarray,
    recon: np.ndarray,
    centerlines: List[List[np.ndarray]],
    city_names: List[str],
    idx: Union[int, None] = None,
    show: bool = True,
    save_path: Union[str, None] = None
) -> None:
    """
    Visualize predicted trajectories alongside ground truth and map lanes.

    Args:
        input_ (np.ndarray): Shape [num_tracks, obs_len, 2], historical trajectory.
        output (List[np.ndarray]): List of K predicted trajectories per track,
                                   each [num_tracks, pred_len, 2].
        target (np.ndarray): Shape [num_tracks, pred_len, 2], ground-truth future.
        centerlines (List[List[np.ndarray]]): Lane polylines for each agent.
        city_names (List[str]): City name per trajectory for map lookup.
        idx (int, optional): Specific index to visualize.
        show (bool): Whether to display the figure.
    """
    assert input_.shape[0] == len(centerlines) == len(city_names)
    print("ploting>>>>>>>>>>")
    # print(len(output[0]))
    num_tracks = input_.shape[0]
    obs_len = input_.shape[1]
    pred_len = target.shape[1]

    plt.clf()  # 💥 clear previous plots
    plt.figure(figsize=(12, 10))   # don't reuse old figure ID
    avm = ArgoverseMap()

    plt.plot(recon[ :, 0], recon[ :, 1], "--", color="#001AFF", linewidth=2, zorder=15)
    #plt.plot(recon[ -1, 0], recon[ -1, 1], "o", color="#001AFF", markersize=9, zorder=15)

    # Draw arrow pointing to the last point of recon
    x_end, y_end = recon[-1]
    x_prev, y_prev = recon[-2]
    dx, dy = x_end - x_prev, y_end - y_prev

    plt.arrow(
        x_end - dx * 0.5, y_end - dy * 0.5,  # start a bit before the end
        dx * 0.5, dy * 0.5,                  # short arrow segment
        head_width=0.5, head_length=0.8,
        fc="#001AFF", ec="#001AFF",
        linewidth=1.5, zorder=15, length_includes_head=True
    )

    for i in range(num_tracks):
        # Plot observed trajectory
        plt.plot(input_[i, :, 0], input_[i, :, 1],"--", color="#ECA154", linewidth=2, zorder=15)
        #plt.plot(input_[i, -1, 0], input_[i, -1, 1], "o", color="#ECA154", markersize=9, zorder=15)
        
        # Draw arrow pointing to the last point
        x_end, y_end = input_[i, -1]
        x_prev, y_prev = input_[i, -2]
        dx, dy = x_end - x_prev, y_end - y_prev

        plt.arrow(
            x_end - dx * 0.5, y_end - dy * 0.5,  # start a bit before the end
            dx * 0.5, dy * 0.5,                  # short arrow segment
            head_width=0.5, head_length=0.8,
            fc="#ECA154", ec="#ECA154",
            linewidth=1.5, zorder=15, length_includes_head=True
        )




        # Plot ground truth future
        plt.plot(target[i, :, 0], target[i, :, 1],"--", color="#d33e4c", linewidth=2, zorder=20)
        # plt.plot(target[i, -1, 0], target[i, -1, 1], "o", color="#d33e4c", markersize=9, zorder=20)

        # Plot predicted trajectories
        for j in range(len(output[i])):
            plt.plot(output[i][j][:, 0], output[i][j][:, 1],"--", color="#007672", linewidth=2, zorder=15)
            plt.plot(output[i][j][-1, 0], output[i][j][-1, 1], "o", color="#007672", markersize=3, zorder=15)

        # Draw lane centerlines
        # for centerline in centerlines[i]:
        #     plt.plot(centerline[:, 0], centerline[:, 1], "--", color="grey", linewidth=1, zorder=0)

        # Draw lane polygons around observed and predicted trajectories
        for j in range(obs_len):
            lane_ids = avm.get_lane_ids_in_xy_bbox(input_[i, j, 0], input_[i, j, 1], city_names[i], 2.5)
            [avm.draw_lane(lane_id, city_names[i]) for lane_id in lane_ids]
        for j in range(pred_len):
            lane_ids = avm.get_lane_ids_in_xy_bbox(target[i, j, 0], target[i, j, 1], city_names[i], 2.5)
            [avm.draw_lane(lane_id, city_names[i]) for lane_id in lane_ids]

        plt.axis("equal")
        plt.xticks([])
        plt.yticks([])

    # Avoid duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved to: {save_path}")
    if show:
        plt.show()
    plt.close()  # Close the figure to free memory
    print("ploting done>>>>>>>>>>")