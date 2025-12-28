# python
#!/usr/bin/env python
# coding=utf-8
import argparse
import os
import gc
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler
)
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import random
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="ControlNet Visualization Tool")
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--controlnet_path", type=str, required=True)
    parser.add_argument("--rgb_input_dir", type=str, required=True)
    parser.add_argument("--optical_flow_input_dir", type=str, required=True)
    parser.add_argument("--event_input_dir", type=str, default="")
    parser.add_argument("--merged", action="store_true")
    parser.add_argument("--of_norm_factor", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--of_scales", type=str, nargs="+", default=[1])
    parser.add_argument("--prompts", type=str, nargs="+", default=[
        "convert to event frame using Accumulation going forward method",
        # "convert to event frame using Accumulation going reverse method",
    ])
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--flow_max", type=int, default=-1)
    parser.add_argument("--enable_xformers", action="store_true")
    parser.add_argument("--save_result_only", action="store_true")
    parser.add_argument("--binarize", action="store_true", help="Binarize generated outputs (per-channel RGB)")
    parser.add_argument("--binarize_threshold", type=int, default=127, help="Threshold for binarization (0-255)")
    return parser.parse_args()

def preprocess_image(image_path, resolution=512):
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def binarize_array(img_rgb, threshold=127):
    """
    img_rgb: numpy array in RGB with values in 0-255 (uint8) or floats.
    Returns uint8 binary image (0 or 255) per channel.
    """
    arr = img_rgb
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if arr.dtype != np.uint8:
        # if in [0,1], scale up
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return np.where(arr > threshold, 255, 0).astype(np.uint8)

def visualize(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None

    print(f"Loading ControlNet from {args.controlnet_path}")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=weight_dtype)

    print(f"Loading base model from {args.pretrained_model_path}")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae", torch_dtype=weight_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer", use_fast=False)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_path,
        controlnet=controlnet,
        vae=vae,
        tokenizer=tokenizer,
        safety_checker=None,
        torch_dtype=weight_dtype
    ).to(device)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

    if args.enable_xformers:
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory efficient attention")
        except:
            print("Warning: xformers not available. Continuing without it.")

    pipeline.set_progress_bar_config(disable=True)

    rgb_files = [f for f in os.listdir(args.rgb_input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    random.shuffle(rgb_files)
    print(f"Found {len(rgb_files)} images to process")

    for file_name in tqdm(rgb_files, desc="Processing images"):
        if args.save_result_only:
            output = os.path.join(args.output_dir, file_name)
            if os.path.exists(output):
                continue

        rgb_path = os.path.join(args.rgb_input_dir, file_name)
        control_image = Image.open(rgb_path).convert("RGB")
        control_image = control_image.resize((args.resolution, args.resolution), Image.BILINEAR)

        rgb_image = cv2.imread(rgb_path)
        if rgb_image is None:
            print(f"Failed to read rgb: {rgb_path}, skipping.")
            continue
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        height, width = rgb_image.shape[:2]
        aspect_ratio = width / height if height != 0 else 1.0
        new_height = args.resolution
        new_width = int(new_height * aspect_ratio)
        # keep original behavior (the original code used new_height twice mistakenly)
        rgb_image = cv2.resize(rgb_image, (new_height, new_height))

        for p_idx, prompt in enumerate(args.prompts):
            if args.event_input_dir:
                # when events_name list is empty original logic used filename->png replacement
                event_path = os.path.join(args.event_input_dir, file_name.replace(".jpg", ".png"))
                event_image = cv2.imread(event_path)
                if event_image is None:
                    print(f"Event image not found: {event_path}, skipping event.")
                    prompt_images = [rgb_image]
                else:
                    event_image = cv2.resize(event_image, (new_height, new_height))
                    event_image = cv2.cvtColor(event_image, cv2.COLOR_BGR2RGB)
                    prompt_images = [event_image, rgb_image]
            else:
                prompt_images = [rgb_image]

            optical_flow_path = os.path.join(args.optical_flow_input_dir, file_name.replace(".jpg", ".npy").replace(".png", ".npy"))
            if not os.path.exists(optical_flow_path):
                optical_flow_path = optical_flow_path.replace(".npy", "-1.npy")
                if not os.path.exists(optical_flow_path):
                    optical_flow_path = optical_flow_path.replace(".npy", "-flow_up.npy")
                    if not os.path.exists(optical_flow_path):
                        print(f"Optical flow file not found: {optical_flow_path}, skipping.")
                        continue
            optical_flow = np.load(optical_flow_path).squeeze()
            flow_max = np.max(np.abs(optical_flow))
            if args.flow_max != -1 and flow_max > 0:
                rescale_factor = args.flow_max * 0.8 / flow_max
                optical_flow = optical_flow * rescale_factor
            if optical_flow.shape[0] < 5:
                optical_flow = optical_flow.transpose(1, 2, 0)
            of_h, of_w = optical_flow.shape[:2]
            optical_flow = cv2.resize(optical_flow, (args.resolution, args.resolution))
            optical_flow = flow_rescale(optical_flow, (args.resolution, args.resolution), (of_w, of_h))
            optical_flow = optical_flow / args.of_norm_factor
            RGB_im = np.array(control_image) / 255.0

            for scale in args.of_scales:
                optical_flow_scaled = optical_flow * float(scale)
                merged_image = np.concatenate([RGB_im, optical_flow_scaled], axis=2).transpose(2, 0, 1)
                merged_image = torch.from_numpy(merged_image).unsqueeze(0).to(device=device, dtype=weight_dtype)

                if args.save_result_only:
                    annotated_path = os.path.join(
                        args.output_dir,
                        file_name.split(".")[0] + "---" + "scale_" + str(scale) + "---" + prompt.replace(" ", "_") + ".jpg"
                    )
                    if os.path.exists(annotated_path):
                        print("Result already exists, skipping:", annotated_path)
                        continue
                    with torch.autocast(device):
                        output = pipeline(
                            prompt=prompt,
                            image=merged_image,
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=args.guidance_scale,
                            generator=generator,
                        )

                    try:
                        full_vis = np.asarray(output[0])[0]
                        if args.binarize:
                            full_vis = binarize_array(full_vis, threshold=args.binarize_threshold)
                        cv2.imwrite(annotated_path, cv2.cvtColor(full_vis, cv2.COLOR_RGB2BGR))
                    except Exception:
                        PIL_image = output[0][0]
                        if args.binarize:
                            PIL_image = Image.fromarray(binarize_array(np.array(PIL_image), threshold=args.binarize_threshold))
                        PIL_image.save(annotated_path)
                else:
                    if args.merged:
                        out = os.path.join(args.output_dir, file_name.split(".")[0] + "---scale-{}---".format(scale) + prompt.replace(" ", "_") + ".jpg")
                    else:
                        out = os.path.join(args.output_dir, file_name.split(".")[0] + "---scale-{}---".format(scale) + prompt.replace(" ", "_"))
                    if os.path.exists(out):
                        print("Result already exists, skipping:", out)
                        continue
                    for i in range(args.repeat):
                        with torch.autocast(device):
                            output = pipeline(
                                prompt=prompt,
                                image=merged_image,
                                num_inference_steps=args.num_inference_steps,
                                guidance_scale=args.guidance_scale,
                                generator=generator,
                            )
                        try:
                            prompt_images.append(np.asarray(output[0])[0])
                        except:
                            prompt_images.append(np.asarray(output[0][0]))

                    if args.merged:
                        concatenated_image = np.hstack(prompt_images)
                        w = concatenated_image.shape[1]
                        prompt_image = np.ones((100, w, 3), dtype=np.uint8) * 255
                        cv2.putText(prompt_image, f"Prompt: {prompt}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                        full_vis = np.vstack([prompt_image, concatenated_image])
                        if args.binarize:
                            full_vis = binarize_array(full_vis, threshold=args.binarize_threshold)
                        cv2.imwrite(out, cv2.cvtColor(full_vis, cv2.COLOR_RGB2BGR))
                    else:
                        os.makedirs(out, exist_ok=True)
                        if args.event_input_dir:
                            event_img = prompt_images.pop(0)
                            if args.binarize:
                                event_img = binarize_array(event_img, threshold=args.binarize_threshold)
                            cv2.imwrite(os.path.join(out, "event.jpg"), cv2.cvtColor(event_img, cv2.COLOR_RGB2BGR))
                        rgb_img = prompt_images.pop(0)
                        if args.binarize:
                            rgb_img = binarize_array(rgb_img, threshold=args.binarize_threshold)
                        cv2.imwrite(os.path.join(out, "RGB.jpg"), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
                        for idx, prompt_image in enumerate(prompt_images):
                            img_to_save = binarize_array(prompt_image, threshold=args.binarize_threshold) if args.binarize else prompt_image
                            cv2.imwrite(os.path.join(out, f"generated_{idx}.jpg"), cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
                        prompt_images = [rgb_image] if not args.event_input_dir else [event_img, rgb_image]

    del pipeline, controlnet, vae
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"All images processed. Results saved to: {args.output_dir}")

if __name__ == "__main__":
    args = parse_args()
    print(args)
    visualize(args)