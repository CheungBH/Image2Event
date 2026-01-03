import contextlib
import gc
import random
import cv2
import numpy as np
import torch
from pathlib import Path
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler


def flow_rescale(flow, original_size, new_size):
    """Resize flow to new size, scaling the flow values accordingly."""
    zoom_y = new_size[0] / original_size[0]
    zoom_x = new_size[1] / original_size[1]
    flow[..., 0] *= zoom_x
    flow[..., 1] *= zoom_y
    return flow


def load_flow(flowfile: Path, valid_in_3rd_channel: bool):
    assert flowfile.exists()
    assert flowfile.suffix == '.png'

    # imageio reading assumes write format was rgb
    flow_16bit = cv2.imread(str(flowfile), cv2.IMREAD_UNCHANGED)
    flow_16bit = cv2.cvtColor(flow_16bit, cv2.COLOR_BGR2RGB)

    channel3 = flow_16bit[..., -1]
    assert channel3.max() <= 1, f'Maximum value in last channel should be 1: {flowfile}'
    flow, valid2D = flow_16bit_to_float(flow_16bit, valid_in_3rd_channel)
    return flow, valid2D


def flow_16bit_to_float(flow_16bit: np.ndarray, valid_in_3rd_channel: bool):
    assert flow_16bit.dtype == np.uint16
    assert flow_16bit.ndim == 3
    h, w, c = flow_16bit.shape
    assert c == 3

    if valid_in_3rd_channel:
        valid2D = flow_16bit[..., 2] == 1
        assert valid2D.shape == (h, w)
        assert np.all(flow_16bit[~valid2D, -1] == 0)
    else:
        valid2D = np.ones_like(flow_16bit[..., 2], dtype=np.bool_)
    valid_map = np.where(valid2D)

    # to actually compute something useful:
    flow_16bit = flow_16bit.astype('float')

    flow_map = np.zeros((h, w, 2))
    flow_map[valid_map[0], valid_map[1], 0] = (flow_16bit[valid_map[0], valid_map[1], 0] - 2**15) / 128
    flow_map[valid_map[0], valid_map[1], 1] = (flow_16bit[valid_map[0], valid_map[1], 1] - 2**15) / 128
    return flow_map, valid2D




def resize_and_center_crop(img, new_size): # Open the image img = Image.open(image_path)

    # Resize the image to the desired dimensions
    img_resized = img.resize(new_size)

    # Calculate dimensions for center cropping
    width, height = img_resized.size
    target_width, target_height = new_size
    left = (width - target_width) / 2
    top = (height - target_height) / 2
    right = (width + target_width) / 2
    bottom = (height + target_height) / 2

    # Crop the resized image from the center
    cropped_img = img_resized.crop((left, top, right, bottom))

    return cropped_img

def get_vis_sample(dataset, sample_num=8, resolution=512):
    imgs, prompts, targets, flows, warped_images = [], [], [], [], []
    total_samples = len(dataset)
    for _ in range(sample_num):
        i = random.randint(0, total_samples - 1)
        samples = dataset[i]
        imgs.append(resize_and_center_crop(samples['conditioning_image'], (resolution, resolution)))
        prompts.append(samples['text'])
        flows.append(samples['optical_flow'])
        targets.append(resize_and_center_crop(samples['image'], (resolution, resolution)))
        warped_images.append(resize_and_center_crop(samples['warped_image'], (resolution, resolution)))
    return imgs, prompts, targets, flows, warped_images


def visualize(
    prompts, images, flows, warp_images,
    vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, is_final_validation=False, flow_normalize_factor=1.0
):
    # logger.info("Running validation... ")

    if not is_final_validation:
        controlnet = accelerator.unwrap_model(controlnet)
    else:
        controlnet = ControlNetModel.from_pretrained(args.output_dir, torch_dtype=weight_dtype)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)


    image_logs = []
    inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")

    for validation_prompt, validation_image, validation_flow, validation_warped_image in zip(prompts, images, flows, warp_images):
        # validation_image = Image.open(validation_image).convert("RGB")

        images = []

        for _ in range(args.num_validation_images):
            with inference_ctx:
                np_img = np.array(validation_image).astype(np.float32) / 255.0
                # np_img = np_img[None]#.transpose(0, 3, 1, 2)
                # resized_flow = np.resize(validation_flow, (2, 512, 512)).transpose(1, 2, 0)
                resized_flow = validation_flow.astype(np.float32) / flow_normalize_factor
                if args.add_warped_image:
                    np_warped_img = np.array(validation_warped_image).astype(np.float32) / 255.0
                    # np_warped_img = np_warped_img[None]#.transpose(0, 3, 1, 2)
                    merged_input = np.concatenate([np_img, np_warped_img, resized_flow], axis=2)
                else:
                    merged_input = np.concatenate([np_img, resized_flow], axis=2)
                merged_input = merged_input[None].astype(np.float32)
                # tensor_image = torch.from_numpy(np.array(validation_image).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(accelerator.device)
                image = pipeline(
                    validation_prompt, merged_input, num_inference_steps=20, generator=generator
                ).images[0]

            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return image_logs
