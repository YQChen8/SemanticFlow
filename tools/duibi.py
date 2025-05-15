import open3d as o3d
import os
import cv2
import numpy as np


# 创建输出目录
def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


# 渲染单帧点云到图像（无可视化窗口）
def render_frame_with_viewpoint_headless(pcd_path, camera_params, output_image_path):
    # 加载点云
    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        raise ValueError(f"Point cloud at {pcd_path} is empty or failed to load.")

    # 初始化无头模式 Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # 无可视化窗口
    vis.add_geometry(pcd)

    # 设置视角
    ctr = vis.get_view_control()
    if ctr is None:
        raise RuntimeError("Failed to get view control from Visualizer.")
    ctr.convert_from_pinhole_camera_parameters(camera_params)

    # 渲染帧并保存为图像
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_image_path)
    vis.destroy_window()


# 创建视频
def create_video_from_images(image_dir, output_video_path, fps=2):
    images = sorted([img for img in os.listdir(image_dir) if img.endswith(".png")])
    if len(images) == 0:
        print("No images found for video creation.")
        return

    # 获取图像尺寸
    first_image_path = os.path.join(image_dir, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # 创建视频对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 添加帧到视频
    for image in images:
        image_path = os.path.join(image_dir, image)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_video_path}")


# 创建一个默认视角（仅需执行一次）
def create_default_viewpoint(pcd_path, output_viewpoint_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        raise ValueError(f"Point cloud at {pcd_path} is empty or failed to load.")

    # 创建一个默认的相机参数
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # 无窗口
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(output_viewpoint_path, param)
    vis.destroy_window()
    print(f"Default viewpoint saved to {output_viewpoint_path}")


# 主程序
if __name__ == "__main__":
    # 1. 创建默认视角（如果 viewpoint.json 不存在，则生成默认视角）
    reference_pcd = "/home3/hqlab/chenyq/project/SeFlow/output_11label/pc0_105.ply"  # 使用的点云文件路径
    viewpoint_file = "viewpoint.json"
    if not os.path.exists(viewpoint_file):
        print("Creating default viewpoint...")
        create_default_viewpoint(reference_pcd, viewpoint_file)

    # 加载视角参数
    camera_params = o3d.io.read_pinhole_camera_parameters(viewpoint_file)
    if camera_params is None:
        raise ValueError(f"Failed to load viewpoint parameters from {viewpoint_file}")

    # 2. 点云文件路径列表（六个算法的分割结果）
    point_cloud_paths = [
        "/home3/hqlab/chenyq/project/SeFlow/output_11label/pc0_105.ply",
        "/home3/hqlab/chenyq/project/SeFlow/output_11label/pc0_105.ply",
        "/home3/hqlab/chenyq/project/SeFlow/output_11label/pc0_105.ply",
        "/home3/hqlab/chenyq/project/SeFlow/output_11label/pc0_105.ply",
        "/home3/hqlab/chenyq/project/SeFlow/output_11label/pc0_105.ply",
        "/home3/hqlab/chenyq/project/SeFlow/output_11label/pc0_105.ply",
    ]

    # 输出图像和视频路径
    output_dir = "rendered_frames"
    video_output_path = "comparison_video.mp4"
    create_output_dir(output_dir)

    # 3. 渲染每个点云为帧
    for i, pcd_path in enumerate(point_cloud_paths):
        output_image_path = os.path.join(output_dir, f"frame_{i:02d}.png")
        render_frame_with_viewpoint_headless(pcd_path, camera_params, output_image_path)
        print(f"Rendered frame for {pcd_path} saved to {output_image_path}")

    # 4. 创建对比视频
    create_video_from_images(output_dir, video_output_path, fps=2)