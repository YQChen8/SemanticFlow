"""
# Created: 2023-07-17 00:00
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
# 
# This file is part of DeFlow (https://github.com/KTH-RPL/DeFlow) and SeFlow (https://github.com/KTH-RPL/SeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: Define the loss function for training.
"""
import torch
from assets.cuda.chamfer3D import nnChamferDis
MyCUDAChamferDis = nnChamferDis()
from src.utils.av2_eval import CATEGORY_TO_INDEX, BUCKETED_METACATAGORIES
import torch.nn.functional as F
from pointnet2.pointnet2 import *
from torch.cuda.amp import autocast

# NOTE(Qingwen 24/07/06): squared, so it's sqrt(4) = 2m, in 10Hz the vel = 20m/s ~ 72km/h
# If your scenario is different, may need adjust this TRUNCATED to 80-120km/h vel.
TRUNCATED_DIST = 4
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, beta=0.5, reduction='mean'):
        """
        alpha: 控制 Focal Loss 的缩放因子
        gamma: 控制 Focal Loss 的焦点因子
        beta: 前景和背景损失的权重因子，0.5 表示均衡组合
        reduction: 对损失的处理方式
        """
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 转换为二分类标签：0 表示背景，1 表示前景
        binary_targets = (targets != 0).float()

        # 计算前景和背景概率
        probs = torch.softmax(inputs, dim=1)
        bg_prob = probs[:, 0]         # 背景的概率
        fg_prob = 1 - bg_prob         # 前景的总概率

        # 背景的 Focal Loss
        bg_BCE_loss = F.binary_cross_entropy(bg_prob, 1 - binary_targets, reduction='none')
        bg_pt = torch.exp(-bg_BCE_loss)
        bg_focal_loss = self.alpha * (1 - bg_pt) ** self.gamma * bg_BCE_loss

        # 前景的 Focal Loss
        fg_BCE_loss = F.binary_cross_entropy(fg_prob, binary_targets, reduction='none')
        fg_pt = torch.exp(-fg_BCE_loss)
        fg_focal_loss = self.alpha * (1 - fg_pt) ** self.gamma * fg_BCE_loss

        # 加权组合背景和前景的损失
        # bg_ratio = (binary_targets == 0).float().mean()  # 背景比例
        # self.beta = 1 - bg_ratio
        combined_loss = self.beta * fg_focal_loss + (1 - self.beta) * bg_focal_loss

        # 根据 reduction 参数决定返回损失的方式
        if self.reduction == 'mean':
            return combined_loss.mean()
        elif self.reduction == 'sum':
            return combined_loss.sum()
        else:
            return combined_loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算 softmax 得到每个类别的概率
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # 获取每个样本的预测概率
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
# class ForegroundConsistencyLoss(nn.Module):
#     def __init__(self, distance_threshold=0.2):
#         """
#         distance_threshold: 每个类别内前景点之间最大允许的距离，超过该距离的点不考虑一致性损失
#         lambda_consistency: 一致性损失的权重
#         """
#         super(ForegroundConsistencyLoss, self).__init__()
#         self.distance_threshold = distance_threshold

#     def forward(self, point_cloud, pc0_mask):
#         """
#         point_cloud: 点云数据，形状为 [N, 3]，N 为点的数量，3 为点的三维坐标
#         pc0_mask: 网络输出的多分类logits，形状为 [N, C]，N 为点的数量，C 为类别数
#         """
#         # 将 logits 转换为概率，使用 softmax
#         probs = torch.softmax(pc0_mask, dim=1)

#         # 选择所有类别不为 0 的点作为前景点
#         fg_mask = torch.max(probs, dim=1).values > 0.5  # 如果最大类别的概率大于0.5，就认为是前景点
#         foreground_points = point_cloud[fg_mask]
#         foreground_probs = probs[fg_mask]  # 对应的前景点的类别概率

#         if foreground_points.shape[0] == 0:
#             return torch.tensor(0.0).to(point_cloud.device)

#         # 计算每个点的类别
#         fg_labels = torch.argmax(foreground_probs, dim=1)

#         # 存储每个类别的点
#         consistency_loss = 0.0
#         unique_labels = torch.unique(fg_labels)

#         # 遍历每个类别，计算该类别内前景点之间的一致性损失
#         for label in unique_labels:
#             # 获取该类别的前景点
#             label_mask = fg_labels == label
#             label_points = foreground_points[label_mask]

#             if label_points.shape[0] <= 1:
#                 continue  # 如果该类别内的前景点数小于等于1，跳过

#             # 计算前景点之间的距离矩阵
#             distances = torch.cdist(label_points, label_points, p=2)  # 计算欧氏距离

#             # 计算前景点一致性损失：前景点之间的平均距离
#             consistency_loss += torch.sum(distances) / (label_points.shape[0] ** 2)  # 平均距离

#             # 距离超过 threshold 的点不考虑一致性损失
#             consistency_loss += torch.sum(distances < self.distance_threshold) * consistency_loss

#         # 返回加权后的一致性损失
#         return consistency_loss

class ForegroundConsistencyLoss(nn.Module):    
    def __init__(self, max_points=1024, background_class=0, distance_threshold=6.0, min_loss=1.0):
        super(ForegroundConsistencyLoss, self).__init__()
        self.max_points = max_points
        self.background_class = background_class
        self.distance_threshold = distance_threshold
        self.min_loss = min_loss

    def get_foreground_mask(self, logits):
        """获取前景点掩码"""
        _, max_labels = torch.max(logits, dim=1)
        fg_mask = max_labels != self.background_class
        return fg_mask

    def sample_points(self, points, logits, fg_mask):
        """采样前景点到最大数量"""
        fg_points = points[fg_mask]
        fg_logits = logits[fg_mask]

        if fg_points.size(0) > self.max_points:
            indices = torch.randperm(fg_points.size(0), device=fg_points.device)[:self.max_points]
            fg_points = fg_points[indices]
            fg_logits = fg_logits[indices]
        
        return fg_points, fg_logits

    def forward(self, point_cloud, logits):
        """
        point_cloud: 原始点云数据，形状为 [N, 3]
        logits: 每个点的类别概率 logits，形状为 [N, C]
        """
        # 获取前景点并降采样
        fg_mask = self.get_foreground_mask(logits)
        fg_points, fg_logits = self.sample_points(point_cloud, logits, fg_mask)

        # 检查前景点数量是否为0
        num_points = fg_points.size(0)
        if num_points == 0:
            return torch.tensor(self.min_loss, device=point_cloud.device)  # 返回最小损失值

        # 计算前景点之间的距离矩阵
        distances = torch.cdist(fg_points, fg_points, p=2)  # 欧几里得距离
        distant_mask = distances > self.distance_threshold  # 超过距离阈值的点对位置

        # 计算符合条件的 logits 相似度（余弦相似度）
        fg_logits_expanded = fg_logits.unsqueeze(1)  # 扩展维度
        logits_similarity = F.cosine_similarity(fg_logits_expanded, fg_logits_expanded.transpose(0, 1), dim=-1)  # 计算余弦相似度

        # 应用距离掩码，仅对远距离点对施加惩罚
        logits_penalty = logits_similarity[distant_mask].clamp(min=0)

        # 计算损失
        if logits_penalty.numel() > 0:
            loss = torch.mean(logits_penalty)
        else:
            loss = self.min_loss  # 若没有满足条件的点对，返回最小损失值

        return loss
def fit_motion_svd_batch(pc1, pc2, mask=None):
    """
    :param pc1: (B, N, 3) torch.Tensor.
    :param pc2: (B, N, 3) torch.Tensor.
    :param mask: (B, N) torch.Tensor.
    :return:
        R_base: (B, 3, 3) torch.Tensor.
        t_base: (B, 3) torch.Tensor.
    """
    n_batch, n_point, _ = pc1.size()

    if mask is None:
        pc1_mean = torch.mean(pc1, dim=1, keepdim=True)   # (B, 1, 3)
        pc2_mean = torch.mean(pc2, dim=1, keepdim=True)   # (B, 1, 3)
    else:
        pc1_mean = torch.einsum('bnd,bn->bd', pc1, mask) / torch.sum(mask, dim=1, keepdim=True)   # (B, 3)
        pc1_mean.unsqueeze_(1)
        pc2_mean = torch.einsum('bnd,bn->bd', pc2, mask) / torch.sum(mask, dim=1, keepdim=True)
        pc2_mean.unsqueeze_(1)

    pc1_centered = pc1 - pc1_mean
    pc2_centered = pc2 - pc2_mean

    if mask is None:
        S = torch.bmm(pc1_centered.transpose(1, 2), pc2_centered)
    else:
        # 用逐元素乘法替代对角矩阵
        weighted_pc1 = pc1_centered * mask.unsqueeze(-1)
        S = weighted_pc1.transpose(1, 2).bmm(pc2_centered)
        # S = pc1_centered.transpose(1, 2).bmm(torch.diag_embed(mask).bmm(pc2_centered))

    # If mask is not well-defined, S will be ill-posed.
    # We just return an identity matrix.
    valid_batches = ~torch.isnan(S).any(dim=1).any(dim=1)
    R_base = torch.eye(3, device=pc1.device).unsqueeze(0).repeat(n_batch, 1, 1)
    t_base = torch.zeros((n_batch, 3), device=pc1.device)

    if valid_batches.any():
        S = S[valid_batches, ...]
        u, s, v = torch.svd(S, some=False, compute_uv=True)
        R = torch.bmm(v, u.transpose(1, 2))
        det = torch.det(R)

        # Correct reflection matrix to rotation matrix
        diag = torch.ones_like(S[..., 0], requires_grad=False)
        diag[:, 2] = det
        R = v.bmm(torch.diag_embed(diag).bmm(u.transpose(1, 2)))

        pc1_mean, pc2_mean = pc1_mean[valid_batches], pc2_mean[valid_batches]
        t = pc2_mean.squeeze(1) - torch.bmm(R, pc1_mean.transpose(1, 2)).squeeze(2)

        R_base[valid_batches] = R
        t_base[valid_batches] = t

    return R_base, t_base
def rigidLoss(pc, mask, flow):
        # pc = pc.unsqueeze(0)
        # if mask is not None:
        #     mask = mask.unsqueeze(0)
        # flow = flow.unsqueeze(0)
        """
        :param pc: (B, N, 3) torch.Tensor.
        :param mask: (B, N, K) torch.Tensor.
        :param flow: (B, N, 3) torch.Tensor.
        :return:
            loss: () torch.Tensor.
        """

        n_batch, n_point, n_object = mask.size()
        pc2 = pc + flow
        mask = mask.transpose(1, 2).reshape(n_batch * n_object, n_point)
        pc_rep = pc.unsqueeze(1).repeat(1, n_object, 1, 1).reshape(n_batch * n_object, n_point, 3)
        pc2_rep = pc2.unsqueeze(1).repeat(1, n_object, 1, 1).reshape(n_batch * n_object, n_point, 3)

        # Estimate the rigid transformation
        object_R, object_t = fit_motion_svd_batch(pc_rep, pc2_rep, mask)

        # Apply the estimated rigid transformation onto point cloud
        pc_transformed = torch.einsum('bij,bnj->bni', object_R, pc_rep) + object_t.unsqueeze(1).repeat(1, n_point, 1)
        pc_transformed = pc_transformed.reshape(n_batch, n_object, n_point, 3).detach()
        mask = mask.reshape(n_batch, n_object, n_point)

        # Measure the discrepancy of per-point flow
        mask = mask.unsqueeze(-1)
        pc_transformed = (mask * pc_transformed).sum(1)
        loss = (pc_transformed - pc2).norm(p=2, dim=-1)
        return loss.mean()

def KnnLoss(pc, mask, k=32, radius=1, cross_entropy=False, loss_norm=1):
    """
    :param pc: (B, N, 3) torch.Tensor.
    :param mask: (B, N, K) torch.Tensor.
    :return:
        loss: () torch.Tensor.
    """
    mask = mask.permute(0, 2, 1).contiguous()
    dist, idx = knn(k, pc, pc)
    tmp_idx = idx[:, :, 0].unsqueeze(2).repeat(1, 1, k).to(idx.device)
    idx[dist > radius] = tmp_idx[dist > radius]
    nn_mask = grouping_operation(mask, idx.detach())
    if cross_entropy:
        mask = mask.unsqueeze(3).repeat(1, 1, 1, k).detach()
        loss = F.binary_cross_entropy(nn_mask, mask, reduction='none').sum(dim=1).mean(dim=-1)
    else:
        loss = (mask.unsqueeze(3) - nn_mask).norm(p=loss_norm, dim=1).mean(dim=-1)
    return loss.mean()

def BallQLoss(pc, mask, k=64, radius=2, cross_entropy=False, loss_norm=1):
    """
    :param pc: (B, N, 3) torch.Tensor.
    :param mask: (B, N, K) torch.Tensor.
    :return:
        loss: () torch.Tensor.
    """
    mask = mask.permute(0, 2, 1).contiguous()
    idx = ball_query(radius, k, pc, pc)
    nn_mask = grouping_operation(mask, idx.detach())
    if cross_entropy:
        mask = mask.unsqueeze(3).repeat(1, 1, 1, k).detach()
        loss = F.binary_cross_entropy(nn_mask, mask, reduction='none').sum(dim=1).mean(dim=-1)
    else:
        loss = (mask.unsqueeze(3) - nn_mask).norm(p=loss_norm, dim=1).mean(dim=-1)
    return loss.mean()


def SmoothLoss(pc, mask, w_knn=3, w_ball_q=1):
    """
    :param pc: (B, N, 3) torch.Tensor.
    :param mask: (B, N, K) torch.Tensor.
    :return:
        loss: () torch.Tensor.
    """
    loss = (w_knn * KnnLoss(pc, mask)) + (w_ball_q * BallQLoss(pc, mask))
    return loss


def seflowLoss(res_dict, timer=None):
    pc0_mask = res_dict['est_mask0']
    pc0_label = res_dict['pc0_labels']
    # 假设 pc0_mask 是形状为 [N, 64] 的 logits
    # 取每个样本在通道维度上最大值的索引作为标签
    # pc0_label = torch.argmax(pc0_mask, dim=1)  # 输出形状为 [N]
    pc1_label = res_dict['pc1_labels']

    pc0 = res_dict['pc0']
    pc1 = res_dict['pc1']

    est_flow = res_dict['est_flow']

    pseudo_pc1from0 = pc0 + est_flow

    unique_labels = torch.unique(pc0_label)
    pc0_dynamic = pc0[pc0_label > 0]
    pc1_dynamic = pc1[pc1_label > 0]
    # fpc1_dynamic = pseudo_pc1from0[pc0_label > 0]
    # NOTE(Qingwen): since we set THREADS_PER_BLOCK is 256
    have_dynamic_cluster = (pc0_dynamic.shape[0] > 256) & (pc1_dynamic.shape[0] > 256)

    # first item loss: chamfer distance
    # timer[5][1].start("MyCUDAChamferDis")
    # raw: pc0 to pc1, est: pseudo_pc1from0 to pc1, idx means the nearest index
    est_dist0, est_dist1, _, _ = MyCUDAChamferDis.disid_res(pseudo_pc1from0, pc1)
    raw_dist0, raw_dist1, raw_idx0, _ = MyCUDAChamferDis.disid_res(pc0, pc1)
    chamfer_dis = torch.mean(est_dist0[est_dist0 <= TRUNCATED_DIST]) + torch.mean(est_dist1[est_dist1 <= TRUNCATED_DIST])
    # timer[5][1].stop()
    
    # second item loss: dynamic chamfer distance
    # timer[5][2].start("DynamicChamferDistance")
    dynamic_chamfer_dis = torch.tensor(0.0, device=est_flow.device)
    if have_dynamic_cluster:
        dynamic_chamfer_dis += MyCUDAChamferDis(pseudo_pc1from0[pc0_label>0], pc1_dynamic, truncate_dist=TRUNCATED_DIST)
    # timer[5][2].stop()

    # third item loss: exclude static points' flow
    # NOTE(Qingwen): add in the later part on label==0
    static_cluster_loss = torch.tensor(0.0, device=est_flow.device)
    
    # fourth item loss: same label points' flow should be the same
    # timer[5][3].start("SameClusterLoss")
    moved_cluster_loss = torch.tensor(0.0, device=est_flow.device)
    moved_cluster_norms = torch.tensor([], device=est_flow.device)
    for label in unique_labels:
        mask = pc0_label == label
        if label == 0:
            # Eq. 6 in the paper
            static_cluster_loss += torch.linalg.vector_norm(est_flow[mask, :], dim=-1).mean()
        elif label > 0 and have_dynamic_cluster:
            cluster_id_flow = est_flow[mask, :]
            cluster_nnd = raw_dist0[mask]
            if cluster_nnd.shape[0] <= 0:
                continue

            # Eq. 8 in the paper
            sorted_idxs = torch.argsort(cluster_nnd, descending=True)
            nearby_label = pc1_label[raw_idx0[mask][sorted_idxs]] # nonzero means dynamic in label
            non_zero_valid_indices = torch.nonzero(nearby_label > 0)
            if non_zero_valid_indices.shape[0] <= 0:
                continue
            max_idx = sorted_idxs[non_zero_valid_indices.squeeze(1)[0]]
            
            # Eq. 9 in the paper
            max_flow = pc1[raw_idx0[mask][max_idx]] - pc0[mask][max_idx]

            # Eq. 10 in the paper
            moved_cluster_norms = torch.cat((moved_cluster_norms, torch.linalg.vector_norm((cluster_id_flow - max_flow), dim=-1)))
    
    if moved_cluster_norms.shape[0] > 0:
        moved_cluster_loss = moved_cluster_norms.mean() # Eq. 11 in the paper
    elif have_dynamic_cluster:
        moved_cluster_loss = torch.mean(raw_dist0[raw_dist0 <= TRUNCATED_DIST]) + torch.mean(raw_dist1[raw_dist1 <= TRUNCATED_DIST])
    # timer[5][3].stop()

    # 将超出类别范围的标签全部映射到类别 0（表示 "地面" 类别）
    num_classes = pc0_mask.size(1)  # 获取类别数
    pc0_label[pc0_label >= num_classes] = 0
    # 假设 mask 是模型输出的 logits，label 是目标标签
    maskLoss = FocalLoss(alpha=1.0, gamma=2.0)
    mask_loss = maskLoss(pc0_mask, pc0_label.long())
    # 这里计算的是前背景点的分类损失
    BFMaskLoss = BinaryFocalLoss(alpha=1.0, gamma=2.0)
    bf_mask_loss = BFMaskLoss(pc0_mask, pc0_label.long())
    # 这里计算的是前景点的各个类别的一致性损失
    FGConsistentLoss = ForegroundConsistencyLoss()
    fg_consistent = FGConsistentLoss(pc0, pc0_mask)



    # rigid loss 12445016
    pc0 = pc0.unsqueeze(0)
    est_flow = est_flow.unsqueeze(0)
    pc0_mask = pc0_mask.unsqueeze(0)
    rigid_loss = rigidLoss(pc0, pc0_mask, est_flow)
    
    # smooth loss 12445016
    # 确保 N 大于或等于 16384
    target_num_points = 16384
    N = pc0.size(1)
    if N >= target_num_points:
        # 随机选择 16384 个点的索引
        indices = torch.randperm(N)[:target_num_points]
        
        # 使用这些索引对 pc0 和 pc0_mask 进行选择
        pc0_downsampled = pc0[:, indices, :]  # 选择点云数据
        pc0_mask_downsampled = pc0_mask[:, indices, :]  # 选择对应的 mask
    else:
        pc0_downsampled = pc0
        pc0_mask_downsampled = pc0_mask
    
    smooth_loss = SmoothLoss(pc0_downsampled, pc0_mask_downsampled)
    # smooth_loss = torch.tensor(0.0, device=est_flow.device)

    



    res_loss = {
        'chamfer_dis': chamfer_dis,
        'dynamic_chamfer_dis': dynamic_chamfer_dis,
        'static_flow_loss': static_cluster_loss,
        'cluster_based_pc0pc1': moved_cluster_loss,
        'rigid_loss': rigid_loss,
        'smooth_loss': smooth_loss,
        'mask_loss' : mask_loss,
        'bf_mask_loss' : bf_mask_loss,
        'fg_consistent': fg_consistent
    }
    return res_loss

def deflowLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']

    mask_no_nan = (~gt.isnan() & ~pred.isnan() & ~gt.isinf() & ~pred.isinf())
    
    pred = pred[mask_no_nan].reshape(-1, 3)
    gt = gt[mask_no_nan].reshape(-1, 3)

    speed = gt.norm(dim=1, p=2) / 0.1
    # pts_loss = torch.norm(pred - gt, dim=1, p=2)
    pts_loss = torch.linalg.vector_norm(pred - gt, dim=-1)

    weight_loss = 0.0
    speed_0_4 = pts_loss[speed < 0.4].mean()
    speed_mid = pts_loss[(speed >= 0.4) & (speed <= 1.0)].mean()
    speed_1_0 = pts_loss[speed > 1.0].mean()
    if ~speed_1_0.isnan():
        weight_loss += speed_1_0
    if ~speed_0_4.isnan():
        weight_loss += speed_0_4
    if ~speed_mid.isnan():
        weight_loss += speed_mid
    return {'loss': weight_loss}

# ref from zeroflow loss class FastFlow3DDistillationLoss()
def zeroflowLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']
    mask_no_nan = (~gt.isnan() & ~pred.isnan() & ~gt.isinf() & ~pred.isinf())
    
    pred = pred[mask_no_nan].reshape(-1, 3)
    gt = gt[mask_no_nan].reshape(-1, 3)

    error = torch.linalg.vector_norm(pred - gt, dim=-1)
    # gt_speed = torch.norm(gt, dim=1, p=2) * 10.0
    gt_speed = torch.linalg.vector_norm(gt, dim=-1) * 10.0
    
    mins = torch.ones_like(gt_speed) * 0.1
    maxs = torch.ones_like(gt_speed)
    importance_scale = torch.max(mins, torch.min(1.8 * gt_speed - 0.8, maxs))
    # error = torch.norm(pred - gt, dim=1, p=2) * importance_scale
    error = error * importance_scale
    return {'loss': error.mean()}

# ref from zeroflow loss class FastFlow3DSupervisedLoss()
def ff3dLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']
    classes = res_dict['gt_classes']
    # error = torch.norm(pred - gt, dim=1, p=2)
    error = torch.linalg.vector_norm(pred - gt, dim=-1)
    is_foreground_class = (classes > 0) # 0 is background, ref: FOREGROUND_BACKGROUND_BREAKDOWN
    background_scalar = is_foreground_class.float() * 0.9 + 0.1
    error = error * background_scalar
    return {'loss': error.mean()}
