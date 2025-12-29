import os
import argparse
import numpy as np
import torch
import lpips
import cv2
from skimage.metrics import structural_similarity as ssim
from pytorch_fid import fid_score


def calculate_psnr(img1, img2):
    """Calculates the Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_mse(img1, img2):
    """Calculates the Mean Squared Error between two images."""
    # return np.mean((img1 - img2) ** 2)
    return np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))


def calculate_metrics(folder1, folder2, device, target_size=512):
    """
    Calculates and prints PSNR, SSIM, LPIPS, FID, and MSE for images in two folders.
    Matches files based on their names without the extension.

    Args:
        folder1 (str): Path to the first image folder.
        folder2 (str): Path to the second image folder.
        device (torch.device): The device to run LPIPS on ('cuda' or 'cpu').
    """
    if not os.path.isdir(folder1) or not os.path.isdir(folder2):
        print("Error: One or both specified paths are not valid directories.")
        return

    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net='alex').to(device)

    # Get list of images and create a map of basenames to full filenames for folder2
    files1 = sorted(os.listdir(folder1))
    files2_map = {os.path.splitext(f)[0]: f for f in os.listdir(folder2)}

    metrics = {
        "psnr": [],
        "ssim": [],
        "lpips": [],
        "mse": []
    }

    num_pairs = 0
    for filename1 in files1:
        basename1, _ = os.path.splitext(filename1)
        path1 = os.path.join(folder1, filename1)

        if basename1 in files2_map:
            filename2 = files2_map[basename1]
            path2 = os.path.join(folder2, filename2)

            num_pairs += 1
            # Load images using OpenCV
            img1 = cv2.imread(path1)
            img2 = cv2.imread(path2)

            # Ensure images are loaded successfully
            if img1 is None:
                print(f"Warning: Could not read image '{path1}'. Skipping.")
                continue
            if img2 is None:
                print(f"Warning: Could not read image '{path2}'. Skipping.")
                continue

            # Ensure images are the same size
            img1 = cv2.resize(img1, (target_size, target_size))
            img2 = cv2.resize(img2, (target_size, target_size))
            # if img1.shape != img2.shape:
            #     print(f"Warning: Resizing image '{filename2}' to match '{filename1}'.")
            #     img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            # --- Calculate PSNR and MSE ---
            metrics["psnr"].append(calculate_psnr(img1, img2))
            metrics["mse"].append(calculate_mse(img1, img2))

            # --- Calculate SSIM ---
            # For SSIM, it's common to use grayscale images
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            ssim_score, _ = ssim(img1_gray, img2_gray, full=True, data_range=img1_gray.max() - img1_gray.min())
            metrics["ssim"].append(ssim_score)

            # --- Calculate LPIPS ---
            # Convert images to PyTorch tensors
            img1_tensor = torch.from_numpy(img1.transpose(2, 0, 1)).float().to(device) / 255.0 * 2 - 1
            img2_tensor = torch.from_numpy(img2.transpose(2, 0, 1)).float().to(device) / 255.0 * 2 - 1

            with torch.no_grad():
                lpips_score = lpips_model(img1_tensor.unsqueeze(0), img2_tensor.unsqueeze(0)).item()
            metrics["lpips"].append(lpips_score)
        else:
            continue
            print(f"Warning: No match found for '{filename1}' in '{folder2}'.")

    if num_pairs == 0:
        print("No matching image pairs found.")
        return

    # --- Calculate FID ---
    # FID is calculated over the entire dataset, not per image
    print("Calculating FID score...")
    fid_value = fid_score.calculate_fid_given_paths(
        [folder1, folder2],
        batch_size=50,
        device=device,
        dims=2048
    )

    # --- Print Results ---
    print("\n--- Image Quality Metrics ---")
    print(f"Processed {num_pairs} image pairs.")
    print(f"Average PSNR: {np.mean(metrics['psnr']):.4f}")
    print(f"Average SSIM: {np.mean(metrics['ssim']):.4f}")
    print(f"Average LPIPS: {np.mean(metrics['lpips']):.4f}")
    print(f"Average MSE: {np.mean(metrics['mse']):.4f}")
    print(f"FID Score: {fid_value:.4f}")
    print("---------------------------\n")
    avg_metrics = {
        "psnr": np.mean(metrics["psnr"]),
        "ssim": np.mean(metrics["ssim"]),
        "lpips": np.mean(metrics["lpips"]),
        "mse": np.mean(metrics["mse"]),
        "fid": fid_value
    }
    return avg_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate image quality metrics between two folders.")
    parser.add_argument("folder1", type=str, help="Path to the first folder of images (e.g., ground truth).", default="quality_asset/gt")
    parser.add_argument("folder2", type=str, help="Path to the second folder of images (e.g., generated).", default="quality_asset/flux_output")
    parser.add_argument("--target_size", type=int, default=512, help="Target size to resize images for metric calculation.")
    args = parser.parse_args()

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    calculate_metrics(args.folder1, args.folder2, device, target_size=args.target_size)