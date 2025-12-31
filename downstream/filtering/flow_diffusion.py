import cv2
import numpy as np
import os
import tqdm


def calculate_diffusion_score(optical_flow):
    """
    改进的光流扩散评估方法

    Args:
        optical_flow: 光流场, shape (H, W, 2) [dx, dy]

    Returns:
        diffusion_score: 扩散分数 (0-1之间，越接近1表示扩散效果越好)
    """
    H, W = optical_flow.shape[:2]

    # 1. 自动检测扩散中心（而不是固定为图像中心）
    # 通过光流向量的汇聚点估计扩散中心
    magnitude = np.sqrt(optical_flow[:, :, 0] ** 2 + optical_flow[:, :, 1] ** 2)

    # 只考虑足够大的光流向量
    large_flow_mask = magnitude > np.percentile(magnitude, 50)  # 只考虑前50%的光流

    center_x, center_y = W // 2, H // 2

    # 2. 计算径向向量
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    dx_radial = x_coords - center_x
    dy_radial = y_coords - center_y

    # 归一化径向向量
    magnitude_radial = np.sqrt(dx_radial ** 2 + dy_radial ** 2)
    magnitude_radial[magnitude_radial == 0] = 1
    dx_radial_norm = dx_radial / magnitude_radial
    dy_radial_norm = dy_radial / magnitude_radial

    # 3. 归一化光流向量
    magnitude_flow = np.sqrt(optical_flow[:, :, 0] ** 2 + optical_flow[:, :, 1] ** 2)
    magnitude_flow_normalized = magnitude_flow / (np.median(magnitude_flow[magnitude_flow > 0]) + 1e-8)

    # 限制归一化范围，避免极端值
    magnitude_flow_normalized = np.clip(magnitude_flow_normalized, 0, 5)

    # 4. 计算方向一致性（使用更宽松的标准）
    # 归一化光流方向
    magnitude_flow_temp = magnitude_flow.copy()
    magnitude_flow_temp[magnitude_flow_temp == 0] = 1
    dx_flow_norm = optical_flow[:, :, 0] / magnitude_flow_temp
    dy_flow_norm = optical_flow[:, :, 1] / magnitude_flow_temp

    # 计算角度差异（而不是点积）
    dot_products = dx_flow_norm * dx_radial_norm + dy_flow_norm * dy_radial_norm
    angle_differences = np.arccos(np.clip(dot_products, -1, 1))  # 角度差异(弧度)

    # 将角度差异转换为一致性分数（0-1）
    # 30度以内的差异都认为是基本一致的
    max_angle = np.pi / 6  # 30度
    direction_consistency = np.maximum(0, 1 - angle_differences / max_angle)

    # 5. 计算综合分数
    # 结合方向一致性和光流大小
    combined_score = direction_consistency * magnitude_flow_normalized

    # 6. 只考虑有效区域（避免边缘或无效区域影响）
    valid_mask = (magnitude_flow > np.percentile(magnitude_flow, 10))  # 只考虑光流大小在前90%的区域

    if np.sum(valid_mask) > 0:
        final_score = np.mean(combined_score[valid_mask])
    else:
        final_score = np.mean(combined_score)

    # 归一化到0-1范围
    final_score = np.clip(final_score, 0, 1)

    return float(final_score)


if __name__ == '__main__':
    flow_folder = "/home/bhzhang/Documents/code/Image2Event/assets/DSEC_RAFT_single_BDD100k/flow"

    # 初始化累加器和计数器
    total_score = 0.0
    count = 0

    # 遍历所有flow文件
    flow_files = [f for f in os.listdir(flow_folder) if f.endswith('.npy')]
    flow_tqdm = tqdm.tqdm(flow_files, desc="Processing flows")

    for flow_file in flow_tqdm:
        flow_path = os.path.join(flow_folder, flow_file)
        optical_flow = np.load(flow_path)
        score = calculate_diffusion_score(optical_flow)

        total_score += score
        count += 1

        # 可选：实时显示进度
        flow_tqdm.set_postfix({"avg_score": f"{total_score / count:.4f}"})

    # 计算并打印平均值
    if count > 0:
        avg_score = total_score / count
        print(f"\n✅ Average diffusion score: {avg_score:.4f}")
        print(f"📊 Processed {count} flow files")
    else:
        print("❌ No flow files found in the folder")