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


# def prepare_batch(pcd_dir, pose_dir, batch_size=8):
#     """
#     根据点云和位姿路径生成符合网络输入格式的batch。
#     :param pcd_dir: 点云文件目录
#     :param pose_dir: 位姿文件目录
#     :param batch_size: 每批次大小
#     :return: 生成的batch数据
#     """
#     # 获取点云和位姿文件名，并按名称排序
#     pcd_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.pcd')])
#     pose_files = sorted([f for f in os.listdir(pose_dir) if f.endswith('.json')])
    
#     if len(pcd_files) < 2 or len(pose_files) < 2:
#         raise ValueError("点云或位姿文件数量不足，至少需要两帧点云和位姿")
    
#     if len(pcd_files) != len(pose_files):
#         raise ValueError("点云文件与位姿文件数量不匹配")
    
#     batch = []
#     prev_data = None  # 用于存储前一帧的数据
    
#     for i in range(len(pcd_files)):
#         # 构造当前帧点云和位姿文件路径
#         pc0_path = os.path.join(pcd_dir, pcd_files[i])
#         pose0_path = os.path.join(pose_dir, pose_files[i])
        
#         # 加载当前帧的点云和位姿数据
#         pc0 = load_pcd(pc0_path)  # torch.Size([1, N, 3])
#         pose0 = load_pose(pose0_path)  # torch.Size([1, 4, 4])
#         scene_id, timestamp = extract_scene_id_and_timestamp(pc0_path)
        
#         # 初始化当前帧的数据
#         gm0 = torch.zeros_like(pc0)  # 示例：假设gm为全零
#         data_dict = {
#             'scene_id': scene_id,
#             'timestamp': timestamp,
#             'pc0': pc0,
#             'gm0': gm0,
#             'pose0': pose0,
#             'pc1': None,
#             'gm1': None,
#             'pose1': None,
#             'eval_mask': None,  # 示例：可根据需求修改
#         }
        
#         if prev_data:
#             # 更新前一帧的 pc1, gm1, pose1 数据
#             prev_data['pc1'] = data_dict['pc0']
#             prev_data['gm1'] = data_dict['gm0']
#             prev_data['pose1'] = data_dict['pose0']
            
#             # 将完整的前一帧数据加入batch
#             batch.append(prev_data)
        
#         # 当前帧数据保存为 prev_data，用于下一次迭代
#         prev_data = data_dict
        
#         # 如果batch达到指定大小，输出batch
#         if len(batch) == batch_size:
#             yield batch
#             batch = []
    
#     # 处理最后一帧数据（无需填充 pc1, gm1, pose1）
#     if prev_data:
#         batch.append(prev_data)
    
#     # 如果最后还有未输出的batch，输出剩余batch
#     if batch:
#         yield batch
# def prepare_batch(pcd_dir, pose_dir, batch_size=8, device='cuda'):
#     """
#     根据点云和位姿路径生成符合网络输入格式的batch，并将数据放到指定设备上。
#     :param pcd_dir: 点云文件目录
#     :param pose_dir: 位姿文件目录
#     :param batch_size: 每批次大小
#     :param device: 数据放置的设备 ('cuda' 或 'cpu')
#     :return: 生成的batch数据
#     """
#     # 获取点云和位姿文件名，并按名称排序
#     pcd_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.pcd')])
#     pose_files = sorted([f for f in os.listdir(pose_dir) if f.endswith('.json')])
    
#     if len(pcd_files) < 2 or len(pose_files) < 2:
#         raise ValueError("点云或位姿文件数量不足，至少需要两帧点云和位姿")
    
#     if len(pcd_files) != len(pose_files):
#         raise ValueError("点云文件与位姿文件数量不匹配")
    
#     batch = []
#     prev_data = None  # 用于存储前一帧的数据
    
#     for i in range(len(pcd_files)):
#         # 构造当前帧点云和位姿文件路径
#         pc0_path = os.path.join(pcd_dir, pcd_files[i])
#         pose0_path = os.path.join(pose_dir, pose_files[i])
        
#         # 加载当前帧的点云和位姿数据
#         pc0 = load_pcd(pc0_path)  # torch.Size([1, N, 3])
#         pose0 = load_pose(pose0_path)  # torch.Size([1, 4, 4])
#         scene_id, timestamp = extract_scene_id_and_timestamp(pc0_path)
#         # 获取点云的数量 N
#         N = pc0.shape[1]
        
#         # 初始化当前帧的数据
#         gm0 = torch.zeros(1, N, dtype=torch.float32).to(pc0.device)  # 初始化 gm0 为全零，形状为 [1, N]
#         eval_mask = torch.zeros(1, N, dtype=torch.float32).to(pc0.device)  # 初始化 eval_mask 为全零，形状为 [1, N]        
#         # 初始化当前帧的数据
#         gm0 = torch.zeros_like(pc0)  # 示例：假设gm为全零
#         data_dict = {
#             'scene_id': torch.tensor(int(scene_id), device=device).unsqueeze(0),
#             'timestamp': torch.tensor(timestamp, device=device).unsqueeze(0),
#             'pc0': pc0.to(device),  # 将点云数据放到指定设备
#             'gm0': gm0.to(device),  # 假设 gm 需要放到设备上
#             'pose0': pose0.to(device),  # 位姿数据放到设备
#             'pc1': None,
#             'gm1': None,
#             'pose1': None,
#             'eval_mask': eval_mask,  # 示例：可根据需求修改
#         }
        
#         if prev_data:
#             # 更新前一帧的 pc1, gm1, pose1 数据
#             prev_data['pc1'] = data_dict['pc0']
#             prev_data['gm1'] = data_dict['gm0']
#             prev_data['pose1'] = data_dict['pose0']
            
#             # 将完整的前一帧数据加入batch
#             batch.append(prev_data)
        
#         # 当前帧数据保存为 prev_data，用于下一次迭代
#         prev_data = data_dict
        
#         # 如果batch达到指定大小，输出batch
#         if len(batch) == batch_size:
#             yield batch
#             batch = []
    
#     # 处理最后一帧数据（无需填充 pc1, gm1, pose1）
#     if prev_data:
#         batch.append(prev_data)
    
#     # 如果最后还有未输出的batch，输出剩余batch
#     if batch:
#         yield batch
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
        scene_id, timestamp = extract_scene_id_and_timestamp(pc0_path)
        N = pc0.shape[1]

        # 初始化伪标签和掩码
        gm0 = torch.zeros(1, N, dtype=torch.float32).to(pc0.device)
        gm1 = torch.zeros(1, N, dtype=torch.float32).to(pc0.device)
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

@hydra.main(version_base=None, config_path="conf", config_name="suscape")
def main(cfg):
    # 使用示例
    pcd_dir = "/dataset/public_dataset_nas2/sustech_points/suscape_dataset/v1.0-mini-unzipped/lidar/scene-000040/lidar"
    pose_dir = "/dataset/public_dataset_nas2/sustech_points/suscape_dataset/v1.0-mini-unzipped/lidar_pose/scene-000040/lidar_pose"
    pl.seed_everything(cfg.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir
    
    if not os.path.exists(cfg.checkpoint):
        print(f"Checkpoint {cfg.checkpoint} does not exist. Need checkpoints for evaluation.")
        sys.exit(1)
        
    torch_load_ckpt = torch.load(cfg.checkpoint)
    checkpoint_params = DictConfig(torch_load_ckpt["hyper_parameters"])
    cfg.output = checkpoint_params.cfg.output + f"-e{torch_load_ckpt['epoch']}-{cfg.av2_mode}-v{cfg.leaderboard_version}"
    cfg.model.update(checkpoint_params.cfg.model)
 
    mymodel = ModelWrapper.load_from_checkpoint(cfg.checkpoint, cfg=cfg, eval=True).cuda()
    print(f"\n---LOG[eval]: Loaded model from {cfg.checkpoint}. The backbone network is {checkpoint_params.cfg.model.name}.\n")
    for batch in prepare_batch(pcd_dir, pose_dir, batch_size=1):
        batch_test = batch[0]
        for key, value in batch_test.items():
            print(key)
            print(len(value))
            print(f"Key: {key}, Value Type: {type(value)}")
        # # 确保 batch 中所有 Tensor 类型的值都移动到 GPU
        # for key, value in batch_test.items():
        #     if isinstance(value, torch.Tensor):  # 仅对 Tensor 调用 .cuda()
        #         batch_test[key] = value.cuda()
        batch_test, res_dict_test = mymodel.run_model_w_ground_data(batch_test)
        # print(f"Batch size: {len(batch)}")
        # for item in batch:
        #     print(f"Scene ID: {item['scene_id']}, Timestamp: {item['timestamp']}")
        #     print(f"pc0 Shape: {item['pc0'].shape}, pose0 Shape: {item['pose0'].shape}")
        #     if item['pose1'] is not None:
        #         print(f"pc1 Shape: {item['pc1'].shape}, pose1 Shape: {item['pose1'].shape}")
        #     else:
        #         print("pc1: None, pose1: None")
            
        #     batch, res_dict = run_model_w_ground_data(batch)
            

if __name__ == "__main__":
    main()







