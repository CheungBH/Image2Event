
import os
import numpy as np

FLOW_MAX = 200

def compute_metrics(pred, gt):
    # Only consider valid pixels (finite values in both arrays)
    valid = np.isfinite(gt).all(axis=-1) & np.isfinite(pred).all(axis=-1)
    if not np.any(valid):
        return None

    flow_pred = pred[valid]
    flow_gt = gt[valid]

    # EPE
    epe = np.linalg.norm(flow_pred - flow_gt, axis=-1)

    # Magnitude
    mag = np.linalg.norm(flow_gt, axis=-1)

    # F1
    out = ((epe > 3.0) & ((epe / (mag + 1e-8)) > 0.05)).astype(np.float32)

    # Angle
    dot = np.sum(flow_pred * flow_gt, axis=-1)
    norm_pred = np.linalg.norm(flow_pred, axis=-1)
    norm_gt = np.linalg.norm(flow_gt, axis=-1)
    cos_angle = dot / (norm_pred * norm_gt + 1e-8)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle) * 180.0 / np.pi

    return epe.mean(), out.mean() * 100, angle.mean()

def evaluate_folders(pred_folder, gt_folder):
    pred_files = {f for f in os.listdir(pred_folder) if f.endswith('.npy')}
    gt_files = {f.replace(".npy", "-1.npy") for f in os.listdir(gt_folder) if f.endswith('.npy')}
    common_files = pred_files & gt_files

    epe_list, f1_list, angle_list = [], [], []

    for fname in sorted(common_files):
        pred = np.load(os.path.join(pred_folder, fname))
        if FLOW_MAX > 0:
            # pred = pred.numpy()
            abs_max = np.abs(pred).max()
            if abs_max > 0:
                ratio = (FLOW_MAX * 0.8) / abs_max
                pred = pred * ratio

        gt = np.load(os.path.join(gt_folder, fname.replace("-1.npy", ".npy")))[0].transpose((1, 2, 0))
        metrics = compute_metrics(pred, gt)
        if metrics:
            epe, f1, angle = metrics
            epe_list.append(epe)
            f1_list.append(f1)
            angle_list.append(angle)

    print(f"Mean EPE: {np.mean(epe_list):.4f}")
    print(f"Mean F1: {np.mean(f1_list):.2f}")
    print(f"Mean Angle: {np.mean(angle_list):.2f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_folder', required=True, help='Folder with predicted flow .npy files')
    parser.add_argument('--gt_folder', required=True, help='Folder with ground truth flow .npy files')
    args = parser.parse_args()
    evaluate_folders(args.pred_folder, args.gt_folder)