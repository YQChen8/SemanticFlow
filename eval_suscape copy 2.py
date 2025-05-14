import os
import json
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
import hydra, wandb, os, sys
from hydra.core.hydra_config import HydraConfig
from src.dataset import HDF5Dataset
from src.trainer import ModelWrapper
from src.models.basic import cal_pose0to1
import colorsys
# @torch.no_grad()
def run_model_w_ground_data(self, batch):
    # NOTE (Qingwen): only needed when val or test mode, since train we will go through collate_fn to remove.
    batch['origin_pc0'] = batch['pc0'].clone()
    # 移除地面数据的代码已被删除，以保留地面数据
    # batch['pc0'] = batch['pc0']
    # batch['pc1'] = batch['pc1']
    # print(batch['pc0'].shape)
    # exit()
    
    # if 'pcb0' in batch:
        # 同样，移除地面数据的代码已被删除，以保留地面数据
        # batch['pcb0'] = batch['pcb0'][~batch['gmb0']].unsqueeze(0)
        # batch['pcb0'] = batch['pcb0'].unsqueeze(0)

    self.model.timer[12].start("One Scan")
    res_dict = self.model(batch)
    self.model.timer[12].stop()

    # NOTE (Qingwen): Since val and test, we will force set batch_size = 1 
    batch = {key: batch[key][0] for key in batch if len(batch[key])>0}
    res_dict = {key: res_dict[key][0] for key in res_dict if len(res_dict[key])>0}
    return batch, res_dict

def load_pcd(file_path):
    """加载点云数据并返回torch.Tensor格式"""
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    return torch.tensor(points, dtype=torch.float32).unsqueeze(0)  # 转为 torch.Size([1, N, 3])


def load_pose(file_path):
    """加载位姿数据并返回torch.Tensor格式"""
    with open(file_path, 'r') as f:
        data = json.load(f)
        pose_list = data["lidarPose"]
        if len(pose_list) != 16:
            raise ValueError(f"文件 {file_path} 中的位姿数据不是 4x4 矩阵")
        pose = np.array(pose_list, dtype=np.float32).reshape(4, 4)
        return torch.tensor(pose, dtype=torch.float32).unsqueeze(0)  # 转为 torch.Size([1, 4, 4])


def extract_scene_id_and_timestamp(file_path):
    """从文件路径中提取 scene_id 和 timestamp"""
    # 示例路径: /dataset/.../lidar/scene-000040/lidar/1630376256.000.pcd
    parts = file_path.split('/')
    scene_id = parts[-3].split('-')[-1]  # 提取 scene-000040 中的 000040
    timestamp = int(parts[-1].split('.')[0]) * 1000  # 提取 1630376256 转为毫秒
    return scene_id, timestamp
def downsample(point_cloud, factor=0.25):
    """
    对点云进行降采样，保留原始点的指定比例。
    :param point_cloud: 输入点云，形状为 torch.Size([1, N, 3])
    :param factor: 降采样比例，例如 0.5 表示保留一半点数
    :return: 降采样后的点云，形状为 torch.Size([1, M, 3])，其中 M = N * factor
    """
    B, N, C = point_cloud.shape  # 获取批次大小、点数和维度
    M = int(N * factor)  # 降采样后点数

    # 随机选择降采样的点的索引
    sampled_indices = torch.randperm(N)[:M].to(point_cloud.device)

    # 根据索引选择点
    downsampled_pcd = point_cloud[:, sampled_indices, :]
    return downsampled_pcd
def prepare_batch(pcd_dir, pose_dir, batch_size=1, device='cuda'):
    """
    根据点云和位姿路径生成符合网络输入格式的batch，并将数据放到指定设备上。
    确保当前帧和下一帧的数据都不为 None。
    :param pcd_dir: 点云文件目录
    :param pose_dir: 位姿文件目录
    :param batch_size: 每批次大小
    :param device: 数据放置的设备 ('cuda' 或 'cpu')
    :return: 生成的batch数据
    """
    # 获取点云和位姿文件名，并按名称排序
    pcd_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.pcd')])
    pose_files = sorted([f for f in os.listdir(pose_dir) if f.endswith('.json')])

    if len(pcd_files) < 2 or len(pose_files) < 2:
        raise ValueError("点云或位姿文件数量不足，至少需要两帧点云和位姿")

    if len(pcd_files) != len(pose_files):
        raise ValueError("点云文件与位姿文件数量不匹配")

    batch = []
    prev_data = None  # 用于存储前一帧的数据

    for i in range(len(pcd_files) - 1):  # 注意这里少遍历一帧
        # 构造当前帧和下一帧的路径
        pc0_path = os.path.join(pcd_dir, pcd_files[i])
        pose0_path = os.path.join(pose_dir, pose_files[i])
        pc1_path = os.path.join(pcd_dir, pcd_files[i + 1])
        pose1_path = os.path.join(pose_dir, pose_files[i + 1])

        # 加载当前帧和下一帧的点云与位姿数据
        pc0 = load_pcd(pc0_path)  # torch.Size([1, N, 3])
        pose0 = load_pose(pose0_path)  # torch.Size([1, 4, 4])
        pc1 = load_pcd(pc1_path)  # torch.Size([1, N, 3])
        pose1 = load_pose(pose1_path)  # torch.Size([1, 4, 4])
        # 对点云 pc0 和 pc1 降采样到原来的二分之一
        pc0 = downsample(pc0, factor=0.5)
        pc1 = downsample(pc1, factor=0.5)
        scene_id, timestamp = extract_scene_id_and_timestamp(pc0_path)
        N = pc0.shape[1]

        # 初始化伪标签和掩码
        gm0 = torch.zeros(1, N, dtype=torch.bool).to(pc0.device)
        gm1 = torch.zeros(1, N, dtype=torch.bool).to(pc0.device)
        eval_mask = torch.zeros(1, N, dtype=torch.float32).to(pc0.device)

        # 构建数据字典
        data_dict = {
            'scene_id': torch.tensor(int(scene_id), device=device).unsqueeze(0),
            'timestamp': torch.tensor(timestamp, device=device).unsqueeze(0),
            'pc0': pc0.to(device),
            'gm0': gm0.to(device),
            'pose0': pose0.to(device),
            'pc1': pc1.to(device),
            'gm1': gm1.to(device),
            'pose1': pose1.to(device),
            'eval_mask': eval_mask,
        }

        # 将当前数据添加到 batch
        batch.append(data_dict)

        # 输出 batch 数据
        if len(batch) == batch_size:
            yield batch
            batch = []

    # 输出剩余不足一个 batch 的数据
    if batch:
        yield batch
def precheck_cfg_valid(cfg):
    if os.path.exists(cfg.dataset_path + f"/{cfg.av2_mode}") is False:
        raise ValueError(f"Dataset {cfg.dataset_path}/{cfg.av2_mode} does not exist. Please check the path.")
    if cfg.supervised_flag not in [True, False]:
        raise ValueError(f"Supervised flag {cfg.supervised_flag} is not valid. Please set it to True or False.")
    if cfg.leaderboard_version not in [1, 2]:
        raise ValueError(f"Leaderboard version {cfg.leaderboard_version} is not valid. Please set it to 1 or 2.")
    return cfg


# def main(cfg):
#     # 使用示例
#     pcd_dir = "/dataset/public_dataset_nas2/sustech_points/suscape_dataset/v1.0-mini-unzipped/lidar/scene-000040/lidar"
#     pose_dir = "/dataset/public_dataset_nas2/sustech_points/suscape_dataset/v1.0-mini-unzipped/lidar_pose/scene-000040/lidar_pose"
#     pl.seed_everything(cfg.seed, workers=True)
    
#     if not os.path.exists(cfg.checkpoint):
#         print(f"Checkpoint {cfg.checkpoint} does not exist. Need checkpoints for evaluation.")
#         sys.exit(1)
        
#     torch_load_ckpt = torch.load(cfg.checkpoint)
#     checkpoint_params = DictConfig(torch_load_ckpt["hyper_parameters"])
#     cfg.output = checkpoint_params.cfg.output + f"-e{torch_load_ckpt['epoch']}-{cfg.av2_mode}-v{cfg.leaderboard_version}"
#     cfg.model.update(checkpoint_params.cfg.model)
 
#     mymodel = ModelWrapper.load_from_checkpoint(cfg.checkpoint, cfg=cfg, eval=True).cuda()
#     print(f"\n---LOG[eval]: Loaded model from {cfg.checkpoint}. The backbone network is {checkpoint_params.cfg.model.name}.\n")
#     for batch in prepare_batch(pcd_dir, pose_dir, batch_size=1):
#         batch_test = batch[0]
#         batch_test, res_dict_test = mymodel.run_model_w_ground_data(batch_test)
#         pc0 = batch_test['origin_pc0']
#         pose_0to1 = cal_pose0to1(batch_test["pose0"], batch_test["pose1"])
#         transform_pc0 = pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
#         pose_flow = transform_pc0 - pc0

#         final_flow = pose_flow.clone()
#         if 'pc0_valid_point_idxes' in res_dict_test:
#             valid_from_pc2res = res_dict_test['pc0_valid_point_idxes']
#             pred_flow = pose_flow.clone()
#             pred_flow[valid_from_pc2res] = pose_flow[valid_from_pc2res] + res_dict_test['flow']

#             final_flow = pred_flow
#             # else:
#             # # flow in the original pc0 coordinate
#             #     pred_flow = pose_flow[~batch['gm0']].clone()
#             #     pred_flow[valid_from_pc2res] = pose_flow[~batch['gm0']][valid_from_pc2res] + res_dict['flow']

#             #     final_flow[~batch['gm0']] = pred_flow

            
#         else:
#             final_flow[~batch['gm0']] = res_dict_test['flow'] + pose_flow[~batch['gm0']]
#             exit()
#         # write final_flow and final label into the dataset.
#         init_label = torch.zeros(final_flow.shape[0]).to(final_flow.device)
#         final_label = init_label.clone()
#         pred_label = init_label[~batch_test['gm0']].clone()
#         pred_label[valid_from_pc2res] = init_label[~batch_test['gm0']][valid_from_pc2res] + torch.argmax(res_dict_test['masks0'], dim=1).float()
#         final_label[~batch_test['gm0']] = pred_label

def save_pointcloud_with_labels(pc0, final_label, output_dir, file_name):
    """
    保存点云为 PLY 文件，根据标签分配颜色，标签为 0 时设置为灰色。

    Args:
        pc0 (torch.Tensor): 点云数据，形状为 (N, 3)。
        final_label (torch.Tensor): 标签数据，形状为 (N,)。
        output_dir (str): 输出文件夹路径。
        file_name (str): 输出 PLY 文件名。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 确保输入的点云和标签在同一设备上，并转换为 numpy 数组
    pc0_np = pc0.cpu().numpy()  # 点云数据 (N, 3)
    labels_np = final_label.cpu().numpy()  # 标签数据 (N,)

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc0_np)  # 设置点云坐标

    # 获取唯一标签
    unique_labels = np.unique(labels_np)
    num_labels = len(unique_labels)

    # 定义标签颜色映射
    label_colors = {}
    for i, label in enumerate(unique_labels):
        if label == 0:  # 标签为 0 时设为灰色
            label_colors[label] = [0.5, 0.5, 0.5]  # RGB 范围 0-1
        else:
            # 使用 HSV 色彩空间分配颜色
            hue = i / num_labels  # 色相：根据标签数量均匀分布
            saturation = 0.9  # 饱和度
            brightness = 0.8  # 亮度
            rgb = colorsys.hsv_to_rgb(hue, saturation, brightness)  # 转换为 RGB
            label_colors[label] = rgb

    # 根据标签为每个点分配颜色
    colors = np.array([label_colors[label] for label in labels_np])  # 生成颜色数组

    # 设置点云颜色
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 保存为 PLY 文件
    output_path = os.path.join(output_dir, file_name)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"点云已保存到 {output_path}")

@hydra.main(version_base=None, config_path="conf", config_name="suscape")
def main(cfg):
    # 使用示例
    pcd_dir = "/dataset/public_dataset_nas2/sustech_points/suscape_dataset/v1.0-mini-unzipped/lidar/scene-000040/lidar"
    pose_dir = "/dataset/public_dataset_nas2/sustech_points/suscape_dataset/v1.0-mini-unzipped/lidar_pose/scene-000040/lidar_pose"
    output_dir = "output_suscape"
    
    pl.seed_everything(cfg.seed, workers=True)
    
    if not os.path.exists(cfg.checkpoint):
        print(f"Checkpoint {cfg.checkpoint} does not exist. Need checkpoints for evaluation.")
        sys.exit(1)
        
    torch_load_ckpt = torch.load(cfg.checkpoint)
    checkpoint_params = DictConfig(torch_load_ckpt["hyper_parameters"])
    cfg.output = checkpoint_params.cfg.output + f"-e{torch_load_ckpt['epoch']}-{cfg.av2_mode}-v{cfg.leaderboard_version}"
    cfg.model.update(checkpoint_params.cfg.model)
 
    mymodel = ModelWrapper.load_from_checkpoint(cfg.checkpoint, cfg=cfg, eval=True).cuda()
    print(f"\n---LOG[eval]: Loaded model from {cfg.checkpoint}. The backbone network is {checkpoint_params.cfg.model.name}.\n")
    
    for i, batch in enumerate(prepare_batch(pcd_dir, pose_dir, batch_size=1)):
        batch_test = batch[0]
        batch_test, res_dict_test = mymodel.run_model_w_ground_data(batch_test)
        pc0 = batch_test['origin_pc0']
        pose_0to1 = cal_pose0to1(batch_test["pose0"], batch_test["pose1"])
        transform_pc0 = pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
        pose_flow = transform_pc0 - pc0

        final_flow = pose_flow.clone()
        if 'pc0_valid_point_idxes' in res_dict_test:
            valid_from_pc2res = res_dict_test['pc0_valid_point_idxes']
            pred_flow = pose_flow.clone()
            pred_flow[valid_from_pc2res] = pose_flow[valid_from_pc2res] + res_dict_test['flow']

            final_flow = pred_flow
            
        else:
            final_flow[~batch['gm0']] = res_dict_test['flow'] + pose_flow[~batch['gm0']]
            exit()
            
        # Write final_flow and final_label into the dataset.
        init_label = torch.zeros(final_flow.shape[0]).to(final_flow.device)
        final_label = init_label.clone()
        pred_label = init_label[~batch_test['gm0']].clone()
        pred_label[valid_from_pc2res] = init_label[~batch_test['gm0']][valid_from_pc2res] + torch.argmax(res_dict_test['masks0'], dim=1).float()
        final_label[~batch_test['gm0']] = pred_label
        
        # 保存点云到指定文件夹
        file_name = f"scene_{i:04d}.ply"  # 动态生成文件名
        save_pointcloud_with_labels(pc0, final_label, output_dir, file_name)          

if __name__ == "__main__":
    main()







