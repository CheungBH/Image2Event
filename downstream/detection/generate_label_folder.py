import shutil
import os
import tqdm
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default="/home/bhzhang/Documents/datasets/aigc_detection/second_Gen1/synthetic_only/images/train", help='Path to the folder containing images.')
    parser.add_argument('--label_root_folder', type=str, default="/home/bhzhang/Documents/datasets/aigc_detection/second_pipeline/labels_gen1", help='Path to the folder containing labels.')
    parser.add_argument('--label_dest_folder', type=str, default="/home/bhzhang/Documents/datasets/aigc_detection/second_Gen1/synthetic_only/labels/train", help='Path to the destination folder for labels.')
    args = parser.parse_args()

    image_folder = args.image_folder
    label_root_folder = args.label_root_folder
    label_dest_folder = args.label_dest_folder


    os.makedirs(label_dest_folder, exist_ok=True)

    for img_name in tqdm.tqdm(os.listdir(image_folder)):
        label_name = img_name.split("---")[0].split(".")[0] + ".txt"
        label_path = os.path.join(label_root_folder, label_name)
        if os.path.exists(label_path):
            img_name = img_name.replace("jpg", "txt")
            img_name = img_name.replace("png", "txt")
            shutil.copy(label_path, os.path.join(label_dest_folder, img_name))

