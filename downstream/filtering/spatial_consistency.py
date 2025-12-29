import os
import torch
import torch.nn as nn
from torch.hub import load
import cv2
import PIL
from torch.nn import functional as F
import numpy as np
from torchvision import transforms
import tqdm
from collections import defaultdict

# --- Configuration ---
SRC_ROOT = "/home/bhzhang/Documents/visualize/event_images/BDD100k_with_RAFT"
OUTPUT_ROOT = "/media/bhzhang/Crucial/diffusion_project/RAFT_asset/former_checkpoint_for_bdd100k/spatial_image"  # Set to "" to disable image saving
DINO_BACKBONE = 'dinov2_g'  # Options: 'dinov2_s', 'dinov2_b', 'dinov2_l', 'dinov2_g'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- DINOv2 Model Definition ---
dino_backbones = {
    'dinov2_s': {'name': 'dinov2_vits14', 'embedding_size': 384},
    'dinov2_b': {'name': 'dinov2_vitb14', 'embedding_size': 768},
    'dinov2_l': {'name': 'dinov2_vitl14', 'embedding_size': 1024},
    'dinov2_g': {'name': 'dinov2_vitg14', 'embedding_size': 1536},
}


class DinoModel(nn.Module):
    def __init__(self, backbone_name):
        super(DinoModel, self).__init__()
        if backbone_name not in dino_backbones:
            raise ValueError(f"Backbone '{backbone_name}' not recognized.")

        self.backbone = load('facebookresearch/dinov2', dino_backbones[backbone_name]['name'])
        self.backbone.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def forward(self, x):
        return self.backbone(x)


def calculate_average_distance(items, metric='pixel'):
    """Calculates the average distance/similarity between all pairs in a list."""
    if len(items) < 2:
        return 0.0

    distances = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if metric == 'pixel':
                # Mean Absolute Difference for pixels
                dist = np.mean(np.abs(items[i].astype(np.float32) - items[j].astype(np.float32)))
                distances.append(dist)
            elif metric == 'dino':
                # Cosine Similarity for features
                similarity = F.cosine_similarity(items[i].unsqueeze(0), items[j].unsqueeze(0))
                distances.append(similarity.item())
            else:
                raise NotImplementedError

    return sum(distances) / len(distances) if distances else 0.0


def create_report_image(image, scores):
    """Adds a label bar with scores to an image."""
    label_bar_height = 50
    label_bar = np.zeros((label_bar_height, image.shape[1], 3), dtype=np.uint8)

    # Combine scores into a single line of text
    text = f"Pixel Dist: {scores['pixel']:.2f} | DINO RGB Sim: {scores['dino_rgb']:.4f} | DINO Gray Sim: {scores['dino_gray']:.4f}"

    # Position the text on the label bar
    font_scale = 0.8
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    text_y = (label_bar_height + text_height) // 2
    cv2.putText(label_bar, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    return np.vstack([label_bar, image])


# --- Main Execution ---
if __name__ == '__main__':
    if OUTPUT_ROOT:
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        pixel_dist_path = os.path.join(OUTPUT_ROOT, "pixel_dist.txt")
        dino_rgb_path = os.path.join(OUTPUT_ROOT, "dino_rgb.txt")
        dino_gray_path = os.path.join(OUTPUT_ROOT, "dino_gray.txt")
    else:
        pixel_dist_path = "/media/bhzhang/Crucial/diffusion_project/1007/no_control/metric/pixel_dist.txt"
        dino_rgb_path = "/media/bhzhang/Crucial/diffusion_project/1007/no_control/metric/dino_rgb.txt"
        dino_gray_path = "/media/bhzhang/Crucial/diffusion_project/1007/no_control/metric/dino_gray.txt"


    print(f"Using device: {DEVICE}")
    model = DinoModel(DINO_BACKBONE).to(DEVICE)

    subfolders = [f for f in os.listdir(SRC_ROOT) if os.path.isdir(os.path.join(SRC_ROOT, f))]

    with open(pixel_dist_path, 'w') as f_pixel, \
         open(dino_rgb_path, 'w') as f_rgb, \
         open(dino_gray_path, 'w') as f_gray:

        f_pixel.write("name,metric\n")
        f_rgb.write("name,metric\n")
        f_gray.write("name,metric\n")

        for folder in tqdm.tqdm(subfolders, desc="Processing folders"):
            folder_path = os.path.join(SRC_ROOT, folder)

            files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith("generated_")])
            if len(files) != 4:
                # print(f"Warning: Expected 4 'generated_' images in {folder_path}, but found {len(files)}. Skipping.")
                continue

            # --- Load images for all metrics ---
            pixel_images = [cv2.imread(file) for file in files]
            if not pixel_images or any(img is None for img in pixel_images):
                # print(f"Warning: Could not load one or more images in {folder_path}. Skipping.")
                continue

            pil_images_rgb = [PIL.Image.open(file).convert('RGB') for file in files]
            pil_images_gray = [img.convert('L').convert('RGB') for img in pil_images_rgb]  # DINO needs 3 channels

            # --- Calculate Metrics ---
            avg_pixel_dist = calculate_average_distance(pixel_images, metric='pixel')

            rgb_tensors = torch.stack([model.transform(img) for img in pil_images_rgb]).to(DEVICE)
            rgb_features = model(rgb_tensors)
            avg_dino_rgb_sim = calculate_average_distance(rgb_features, metric='dino')

            gray_tensors = torch.stack([model.transform(img) for img in pil_images_gray]).to(DEVICE)
            gray_features = model(gray_tensors)
            avg_dino_gray_sim = calculate_average_distance(gray_features, metric='dino')

            # --- Write results to files ---
            f_pixel.write(f"{folder},{avg_pixel_dist:.4f}\n")
            f_rgb.write(f"{folder},{avg_dino_rgb_sim:.4f}\n")
            f_gray.write(f"{folder},{avg_dino_gray_sim:.4f}\n")

            # --- Create and Save Report Image (if output path is set) ---
            if OUTPUT_ROOT:
                scores = {
                    'pixel': avg_pixel_dist,
                    'dino_rgb': avg_dino_rgb_sim,
                    'dino_gray': avg_dino_gray_sim
                }

                # Arrange the 4 images into a 2x2 grid
                row1 = np.hstack(pixel_images[0:2])
                row2 = np.hstack(pixel_images[2:4])
                grid_image = np.vstack([row1, row2])

                report_image = create_report_image(grid_image, scores)
                output_path = os.path.join(OUTPUT_ROOT, f"{folder}_report.png")
                cv2.imwrite(output_path, report_image)

    print(f"\nProcessing complete. Result files saved.")