"""
# Created: 2023-11-29 21:22
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of DeFlow (https://github.com/KTH-RPL/DeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: Save scene flow dataset after preprocess.
"""

import colorsys
import numpy as np
import fire, time
import open3d as o3d
import os, sys
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
from src.utils.mics import HDF5Data, flow_to_rgb
from src.utils.o3d_view import MyVisualizer, color_map

def save_point_cloud(pcd, filename):
    """Save the point cloud to a file."""
    o3d.io.write_point_cloud(filename, pcd)

def check_flow(
    data_dir: str = "/home/kin/data/av2/preprocess/sensor/mini",
    res_name: str = "flow",  # "flow", "flow_est"
    start_id: int = 0,
    output_dir: str = "output",  # Directory to save point clouds
):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    dataset = HDF5Data(data_dir, vis_name=res_name, flow_view=True)

    for data_id in range(start_id, len(dataset)):
        data = dataset[data_id]
        now_scene_id = data['scene_id']
        print(f"Processing id: {data_id}, scene_id: {now_scene_id}, timestamp: {data['timestamp']}")

        gm0 = data['gm0']
        pc0 = data['pc0'][~gm0]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])
        pcd.paint_uniform_color([1.0, 0.0, 0.0])  # red: pc0

        pc1 = data['pc1']
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pc1[:, :3][~data['gm1']])
        pcd1.paint_uniform_color([0.0, 1.0, 0.0])  # green: pc1

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pc0[:, :3] + data[res_name][~gm0])
        pcd2.paint_uniform_color([0.0, 0.0, 1.0])  # blue: pc0 + flow

        # Save point clouds
        save_point_cloud(pcd, os.path.join(output_dir, f"pc0_{data_id}.ply"))
        save_point_cloud(pcd1, os.path.join(output_dir, f"pc1_{data_id}.ply"))
        save_point_cloud(pcd2, os.path.join(output_dir, f"pc0_flow_{data_id}.ply"))

def vis(
    data_dir: str = "/home/kin/data/av2/preprocess/sensor/mini",
    res_name: str = "flow",  # "flow", "flow_est"
    start_id: int = -1,
    output_dir: str = "output",  # Directory to save point clouds
):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    dataset = HDF5Data(data_dir, vis_name=res_name, flow_view=True)

    for data_id in range(start_id, len(dataset)):
        data = dataset[data_id]
        now_scene_id = data['scene_id']
        print(f"Processing id: {data_id}, scene_id: {now_scene_id}, timestamp: {data['timestamp']}")

        pc0 = data['pc0']
        gm0 = data['gm0']
        pose0 = data['pose0']
        pose1 = data['pose1']
        ego_pose = np.linalg.inv(pose1) @ pose0

        pose_flow = pc0[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0[:, :3]

        pcd = o3d.geometry.PointCloud()
        if res_name in ['dufo_label', 'label']:
            labels = data[res_name]
            pcd_i = o3d.geometry.PointCloud()
            for label_i in np.unique(labels):
                pcd_i.points = o3d.utility.Vector3dVector(pc0[labels == label_i][:, :3])
                if label_i <= 0:
                    pcd_i.paint_uniform_color([1.0, 1.0, 1.0])
                else:
                    pcd_i.paint_uniform_color(color_map[label_i % len(color_map)])
                pcd += pcd_i
        elif 'est_label0' in data and res_name in data:
            label = data['est_label0'].astype(int)
            pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])
            # 获取实际出现的标签
            unique_labels = np.unique(label)
            num_labels = len(unique_labels)

            # 使用HSV空间生成颜色
            label_colors = {}
            for i, lbl in enumerate(unique_labels):
                hue = i / num_labels  # 按标签数量均匀分布色相
                saturation = 0.9  # 设置较高的饱和度
                brightness = 0.8  # 设置较高的亮度
                rgb = colorsys.hsv_to_rgb(hue, saturation, brightness)
                label_colors[lbl] = rgb

            # 设置标签0的颜色为白色
            label_colors[0] = [0.5, 0.5, 0.5]
            # label_colors[50] = [1, 1, 1]

            # 为每个点分配颜色
            colors = np.array([label_colors[lbl] for lbl in label])

            # 设置点云颜色
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # 设置点云颜色
            pcd.colors = o3d.utility.Vector3dVector(colors)
        elif res_name in data:
            pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])
            flow = data[res_name] - pose_flow  # ego motion compensation here.
            flow_color = flow_to_rgb(flow) / 255.0
            is_dynamic = np.linalg.norm(flow, axis=1) > 0.1
            flow_color[~is_dynamic] = [1, 1, 1]
            flow_color[gm0] = [1, 1, 1]
            pcd.colors = o3d.utility.Vector3dVector(flow_color)

        # Save point cloud
        save_point_cloud(pcd, os.path.join(output_dir, f"pc0_{data_id}.ply"))

if __name__ == '__main__':
    start_time = time.time()
    # fire.Fire(check_flow)
    fire.Fire(vis)
    print(f"Time used: {time.time() - start_time:.2f} s")