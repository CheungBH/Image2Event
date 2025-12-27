import sys

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import tqdm

from RAFT.raft import RAFT
from RAFT.utils import flow_viz
from RAFT.utils.utils import InputPadder

DEVICE = 'cuda'


def visualize_optical_flow(flow, frame):
    """Generates a visualization of the optical flow field."""
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
    return vis

def flow_to_image(flow, max_flow=10.0):
    """
    将光流数据转换为可视化图像（颜色过渡更平滑）

    参数:
        flow (np.ndarray): 光流数据，形状 (H, W, 2)
        max_flow (float): 最大光流值，用于归一化（默认10.0）

    返回:
        np.ndarray: BGR 格式的可视化图像
    """
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    magnitude = np.clip(magnitude / max_flow, 0, 1)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)

    hue = (angle % 360) / 2  # 0-180度范围
    hsv[..., 0] = np.clip(hue, 0, 179).astype(np.uint8)
    hsv[..., 1] = magnitude * 255  # 饱和度表示速度
    hsv[..., 2] = 255  # 明度

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def add_label(image, label, color):
    img = image.copy()
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    return img


def merge_images(input_folders, output_folder, notes, colors):
    """
    Merges images from multiple input folders into a single output folder,
    creating a 2x2 merged image for each set of four images
    (one from each input folder).

    Args:
        input_folders: A list of paths toad the four input folders.
        output_folder: The path to the output folder where merged images will be saved.
    """

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the list of image files from each input folder
    image_files = []
    for folder in input_folders:
        img_files = sorted([f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        image_files.append(img_files)

    # Determine the number of sets of images we can create
    num_sets = min(len(files) for files in image_files)

    for i in range(num_sets):
        # Read the i-th image from each folder
        images = []
        for j, folder in enumerate(input_folders):
            image_path = os.path.join(folder, image_files[j][i])
            img = cv2.imread(image_path)

            if img is None:
                print(f"Error: Could not read image {image_path}")
                return  # Exit if any image fails to load
            # img = add_label(img, notes[j], colors[j])
            images.append(img)

        # Ensure all images have the same shape as the first image
        height, width, channels = images[0].shape
        for k in range(1, len(images)):
            images[k] = cv2.resize(images[k], (width, height))

        # Create the 2x2 merged image
        # top_row = np.hstack(])
        merged_image = np.vstack([images[0], images[1], images[2]])
        # merged_image = np.vstack([top_row, bottom_row])

        base_file_name = os.path.splitext(image_files[0][i])[0]
        # Save the merged image
        output_path = os.path.join(output_folder, f"{base_file_name}.png")
        cv2.imwrite(output_path, merged_image)
        print(f"Merged image saved to {output_path}")
        cv2.imshow("Merged image", merged_image)
        cv2.waitKey(1)




def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def save_image(img_tensor, out_path):
    img = img_tensor[0].permute(1,2,0).cpu().numpy().astype(np.uint8)
    cv2.imwrite(out_path, img[:, :, [2,1,0]])  # Convert RGB to BGR for OpenCV

def save_flow(flow_tensor, out_path):
    flo = flow_tensor[0].permute(1,2,0).cpu().numpy()
    flo_img = flow_viz.flow_to_image(flo)
    cv2.imwrite(out_path, flo_img[:, :, [2,1,0]])  # Convert RGB to BGR

def warp_image(image, flow):
    # image: torch tensor [1,3,H,W], flow: torch tensor [1,2,H,W]
    img = image[0].permute(1,2,0).cpu().numpy().astype(np.uint8)
    flo = flow[0].permute(1,2,0).cpu().numpy()
    h, w = flo.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flo[..., 0]).astype(np.float32)
    map_y = (grid_y + flo[..., 1]).astype(np.float32)
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return warped

def demo(args, original_image_folder, flow_folder, warped_image_folder, frame_folder, scale=1.0):

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(original_image_folder, '*.png')) + \
                 glob.glob(os.path.join(original_image_folder, '*.jpg'))
        images = sorted(images)
        for imfile in tqdm.tqdm(images):
            image1 = load_image(imfile)
            image2 = load_image(imfile)


            padder = InputPadder(image1.shape)
            image1_pad, image2_pad = padder.pad(image1, image2)

            flow_low, flow_up = model(image1_pad, image2_pad, iters=20, test_mode=True)

            # Unpad flow to original size
            flow_up_unpad = padder.unpad(flow_up)

            flow_01 = flow_up_unpad[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
            flow_16bit = cv2.cvtColor(
                np.concatenate((flow_01 * 64. + (2 ** 15), np.ones_like(flow_01)[:, :, 0:1]), -1),
                cv2.COLOR_BGR2RGB
            )
            base_name = os.path.splitext(os.path.basename(imfile))[0]
            flow_img_path = os.path.join(flow_folder, f'{base_name}.png')
            cv2.imwrite(flow_img_path, flow_16bit.astype(np.uint16))
            flow_up_unpad = flow_up_unpad * scale  # Scale the flow

            # Warp image1 to synthesize image2
            warped = warp_image(image1, flow_up_unpad)

            # Use original image1 name (without extension) for saving
            base_name = os.path.splitext(os.path.basename(imfile))[0]
            warped_path = os.path.join(warped_image_folder, f'{base_name}.png')
            flow_path = os.path.join(flow_folder, f'{base_name}.npy')

            cv2.imwrite(warped_path, cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
            np.save(flow_path, flow_up_unpad[0].permute(1, 2, 0).cpu().numpy())
            save_image(image1, os.path.join(frame_folder, f'{base_name}.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")

    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument("--input_folder", type=str, help="path to the input image folder", default='datasets/bdd100k/RGB')
    parser.add_argument("--output_folder", type=str, help="path to the input image folder", default='datasets/bdd100k')

    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    original_image_root = args.input_folder
    output_root = args.output_folder
    flow_root = os.path.join(output_root, 'flow')
    warped_image_root = os.path.join(output_root, 'warped')
    frame_root = os.path.join(output_root, 'frame')
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    notes = ["first frame", "warped", "optical"]


    os.makedirs(flow_root, exist_ok=True)
    os.makedirs(warped_image_root, exist_ok=True)
    os.makedirs(frame_root, exist_ok=True)
    demo(args, original_image_root, flow_root, warped_image_root, frame_root, scale=1.0)
    out_merged_path = os.path.join(output_root, 'merged')
    inp_folders = [frame_root, warped_image_root, flow_root]
    merge_images(inp_folders, out_merged_path, notes, colors)
