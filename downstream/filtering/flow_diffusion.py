import cv2
import numpy as np
import os
import tqdm


def calculate_diffusion_score(optical_flow):
    """
    Improved optical flow diffusion evaluation method.

    Args:
        optical_flow: Optical flow field, shape (H, W, 2) [dx, dy]

    Returns:
        diffusion_score: Diffusion score (between 0-1, closer to 1 indicates better diffusion effect)
    """
    H, W = optical_flow.shape[:2]

    # 1. Diffusion Center Determination Strategy (Three-Stage Rocket Mode)
    center_x, center_y = W // 2, H // 2  # Default: Geometric center

    # Preprocessing: Calculate flow magnitude and active mask
    magnitude = np.sqrt(optical_flow[:, :, 0] ** 2 + optical_flow[:, :, 1] ** 2)
    # Filter out points with significant motion (exclude static background and small noise)
    active_flow_mask = magnitude > 1.0 
    
    if np.sum(active_flow_mask) > 100: # At least some points are moving, otherwise calculation is impossible
        active_flow_x = optical_flow[:, :, 0][active_flow_mask]
        
        # --- Stage 1: Sign Consistency Voting (Turn Detection) ---
        # Count the ratio of points moving left and right
        ratio_right = np.sum(active_flow_x > 0.5) / len(active_flow_x)
        ratio_left = np.sum(active_flow_x < -0.5) / len(active_flow_x)
        
        # Only consider it a turn if the vast majority (>70%) are moving in one direction
        is_turning = max(ratio_right, ratio_left) > 0.7
        
        if is_turning:
            try:
                # --- Stage 2: Least Squares FOE Estimation ---
                # Construct equation v*x0 - u*y0 = v*x - u*y
                # Ax = b, x = [x0, y0]
                
                # For calculation stability, use only the top 50% strongest points
                strong_mask = magnitude > np.percentile(magnitude, 50)
                y_grid, x_grid = np.mgrid[0:H, 0:W]
                
                u = optical_flow[:, :, 0][strong_mask]
                v = optical_flow[:, :, 1][strong_mask]
                x = x_grid[strong_mask]
                y = y_grid[strong_mask]
                
                # A = [v, -u]
                A = np.column_stack((v, -u))
                # b = v*x - u*y
                b = v * x - u * y
                
                # Solve
                res = np.linalg.lstsq(A, b, rcond=None)
                est_x, est_y = res[0]
                
                # --- Stage 3: Sanity Check (Fallback Mechanism) ---
                # Check 1: Vertical deviation should not be too large (cars don't pitch violently)
                # Allow deviation of 30% of image height
                valid_y = abs(est_y - H/2) < (H * 0.3)
                
                # Check 2: Horizontal deviation allowed, but not numerical explosion (e.g., NaN or infinite)
                valid_x = np.isfinite(est_x) and abs(est_x) < W * 5 # Limit within 5x width
                
                if valid_y and valid_x:
                    center_x, center_y = est_x, est_y
                    # print(f"Turn detected! Adjusted FOE to ({center_x:.1f}, {center_y:.1f})")
                else:
                    # print(f"Turn detected but FOE unstable ({est_x:.1f}, {est_y:.1f}), fallback to center")
                    pass
                    
            except Exception:
                # Solver failed, fallback
                pass

    # 2. Calculate Radial Vectors
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    dx_radial = x_coords - center_x
    dy_radial = y_coords - center_y

    # Normalize radial vectors
    magnitude_radial = np.sqrt(dx_radial ** 2 + dy_radial ** 2)
    magnitude_radial[magnitude_radial == 0] = 1
    dx_radial_norm = dx_radial / magnitude_radial
    dy_radial_norm = dy_radial / magnitude_radial

    # 3. Normalize Optical Flow Vectors
    magnitude_flow = np.sqrt(optical_flow[:, :, 0] ** 2 + optical_flow[:, :, 1] ** 2)
    magnitude_flow_normalized = magnitude_flow / (np.median(magnitude_flow[magnitude_flow > 0]) + 1e-8)

    # Limit normalization range to avoid extreme values
    # Autonomous driving scenario: Set to 2.0, rejecting heroism.
    # This means any single point can contribute at most 2x the mean score.
    # Only when a large area (>50%) performs well can the total score approach 1.
    magnitude_flow_normalized = np.clip(magnitude_flow_normalized, 0, 2.0)

    # 4. Calculate Direction Consistency (using looser criteria)
    # Normalize flow direction
    magnitude_flow_temp = magnitude_flow.copy()
    magnitude_flow_temp[magnitude_flow_temp == 0] = 1
    dx_flow_norm = optical_flow[:, :, 0] / magnitude_flow_temp
    dy_flow_norm = optical_flow[:, :, 1] / magnitude_flow_temp

    # Calculate angle difference (instead of dot product)
    dot_products = dx_flow_norm * dx_radial_norm + dy_flow_norm * dy_radial_norm
    angle_differences = np.arccos(np.clip(dot_products, -1, 1))  # Angle difference (radians)

    # Convert angle difference to consistency score (0-1)
    # Differences within 30 degrees are considered consistent
    max_angle = np.pi / 6  # 30 degrees
    direction_consistency = np.maximum(0, 1 - angle_differences / max_angle)

    # 5. Calculate Combined Score
    # Combine direction consistency and flow magnitude
    combined_score = direction_consistency * magnitude_flow_normalized

    # 6. Consider Only Valid Regions (avoid edge or invalid region effects)
    valid_mask = (magnitude_flow > np.percentile(magnitude_flow, 10))  # Only consider top 90% magnitude regions

    # --- New: Autonomous Driving Hood Masking ---
    # Force bottom 15% to be invalid to prevent static hood flow from lowering score
    hood_height = int(H * 0.15)
    if hood_height > 0:
        valid_mask[-hood_height:, :] = False

    if np.sum(valid_mask) > 0:
        final_score = np.mean(combined_score[valid_mask])
    else:
        final_score = np.mean(combined_score)

    # Normalize to 0-1 range
    final_score = np.clip(final_score, 0, 1)

    return float(final_score)


if __name__ == '__main__':
    flow_folder = "/home/bhzhang/Documents/code/Image2Event/assets/DSEC_RAFT_single_BDD100k/flow"

    total_score = 0.0
    count = 0

    flow_files = [f for f in os.listdir(flow_folder) if f.endswith('.npy')]
    flow_tqdm = tqdm.tqdm(flow_files, desc="Processing flows")

    for flow_file in flow_tqdm:
        flow_path = os.path.join(flow_folder, flow_file)
        optical_flow = np.load(flow_path)
        score = calculate_diffusion_score(optical_flow)

        total_score += score
        count += 1

        flow_tqdm.set_postfix({"avg_score": f"{total_score / count:.4f}"})

    if count > 0:
        avg_score = total_score / count
        print(f"\n✅ Average diffusion score: {avg_score:.4f}")
        print(f"📊 Processed {count} flow files")
    else:
        print("❌ No flow files found in the folder")