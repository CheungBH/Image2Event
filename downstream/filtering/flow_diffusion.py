import cv2
import numpy as np
import os
import tqdm

def visualize_optical_flow(flow, frame, window_name="Optical Flow"):
    h, w = frame.shape[:2]
    step = 16
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    # cv2.imshow(window_name, vis)
    return vis


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

def calculate_hole_ratio_point_mapping(flow):
    h, w = flow.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.round(grid_x + flow[..., 0]).astype(int)
    map_y = np.round(grid_y + flow[..., 1]).astype(int)
    valid = (map_x >= 0) & (map_x < w) & (map_y >= 0) & (map_y < h)
    mask[map_y[valid], map_x[valid]] = True
    hole_ratio = 1.0 - np.sum(mask) / (h * w)
    return hole_ratio

if __name__ == '__main__':
    flow_folder = "/media/bhzhang/Crucial/diffusion_project/RAFT_asset/former_checkpoint_for_bdd100k/flow"
    display_img_folder = ""
    warped_folder = ""
    output_folder = ""
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    scores_file_path = "/media/bhzhang/Crucial/diffusion_project/RAFT_asset/former_checkpoint_for_bdd100k/hole_scores.txt"
    ratios_file_path = "/media/bhzhang/Crucial/diffusion_project/RAFT_asset/former_checkpoint_for_bdd100k/diffusion_ratios.txt"

    flow_tqdm = tqdm.tqdm(os.listdir(flow_folder))
    with open(scores_file_path, 'w') as scores_file, open(ratios_file_path, 'w') as ratios_file:
        scores_file.write("name,score\n")
        ratios_file.write("name,ratio\n")

        for flow_file in flow_tqdm:
            if not flow_file.endswith('.npy'):
                continue

            flow_path = os.path.join(flow_folder, flow_file)
            optical_flow = np.load(flow_path)
            score = calculate_diffusion_score(optical_flow)
            hole_ratio = calculate_hole_ratio_point_mapping(optical_flow)

            file_name = flow_file.replace(".npy", "")
            scores_file.write(f"{file_name},{score:.4f}\n")
            ratios_file.write(f"{file_name},{hole_ratio:.4f}\n")

            score_str = f"Diffusion score: {score:.4f}; Hole ratio: {hole_ratio:.4f}"
            # print(f"{flow_file}: {score_str}")
            if output_folder:
                image_path = os.path.join(display_img_folder, flow_file.replace(".npy", ".png"))
                raw_img = cv2.imread(image_path)
                if raw_img is None:
                    print(f"Warning: Could not read image {image_path}. Skipping visualization for {flow_file}.")
                    continue

                h, w = raw_img.shape[:2]
                flow_vis_im = visualize_optical_flow(optical_flow, raw_img)
                grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
                map_x = (grid_x + optical_flow[..., 0]).astype(np.float32)
                map_y = (grid_y + optical_flow[..., 1]).astype(np.float32)

                warped_image = cv2.remap(raw_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))
                label_image = np.zeros((50, flow_vis_im.shape[1], 3), dtype=np.uint8) + 255
                cv2.putText(label_image, score_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                merged_img = cv2.vconcat([raw_img, flow_vis_im, label_image])

                cv2.imwrite(os.path.join(output_folder, flow_file.replace(".npy", ".png")), merged_img)