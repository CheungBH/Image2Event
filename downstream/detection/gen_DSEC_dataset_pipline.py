import os, sys

root = "/home/bhzhang/Documents/datasets/aigc_detection/second"
src_image_folder = os.path.join(root, "synthetic_only/images/train")
src_label_folder = os.path.join(root, "synthetic_only/labels/train")
gen_train_cmd = "python gen_yolo_txt.py --folder_path {}".format(src_image_folder)
os.system(gen_train_cmd)

synthetic_only = False

label_root = "/home/bhzhang/Documents/datasets/DSEC_detection_data/bdd100k/yolo_labels/train"
DSEC_root = "/home/bhzhang/Documents/datasets/DSEC_dataset_1004/DSEC_baseline"
copy_names = ["synthetic_1few", "synthetic_5few", "synthetic_full"]

gen_label_cmd = "python generate_label_folder.py --image_folder {} --label_root_folder {} --label_dest_folder {}".format(
    src_image_folder,
    label_root,
    src_label_folder
)
print(gen_label_cmd)
os.system(gen_label_cmd)

src_label_13cls_folder = os.path.join(root, "synthetic_only/labels_13cls/train")
os.makedirs(src_label_13cls_folder, exist_ok=True)
gen_label_cmd = "python generate_label_folder.py --image_folder {} --label_root_folder {} --label_dest_folder {}".format(
    src_image_folder,
    label_root.replace("yolo_labels", "yolo_labels_allclasses"),
    src_label_13cls_folder
)
os.system(gen_label_cmd)
print("Done generating label folders.")


DSEC_full_folder = os.path.join(DSEC_root, "full")
DSEC_to_copy_paths = ["val.txt", "data.yaml", "images/val", "labels/val", "labels_13cls/val"]
for path in DSEC_to_copy_paths:
    full_path = os.path.join(DSEC_full_folder, path)
    dest_path = os.path.join(root, "synthetic_only", path)
    if os.path.exists(full_path):
        if os.path.isdir(full_path):
            os.system(f"cp -r {full_path} {dest_path}")
        else:
            os.system(f"cp {full_path} {dest_path}")

print("Done copying DSEC assets to folders.")

if synthetic_only:
    cmd = "python gen_yolo_txt.py --folder_path {}".format(src_image_folder)
    os.system(cmd)
    print("Done generating train.txt for synthetic only folder.")
    sys.exit(0)

for name in copy_names:
    src_path = os.path.join(root, "synthetic_only")
    dest_path = os.path.join(root, name)
    os.system(f"cp -r {src_path} {dest_path}")
print("Done copying datasets to different settings.")


train_names = ["1_few", "5_few", "full"]
file_to_copy = ["images/train", "labels/train"]
for src_folder, dest_folder in zip(train_names, copy_names):
    for file in file_to_copy:
        src_path = os.path.join(DSEC_root, src_folder, file)
        dest_path = os.path.join(root, dest_folder, file)
        if file == "labels_13cls/train":
            os.makedirs(dest_path, exist_ok=True)
        for f in os.listdir(src_path):
            src_file_path = os.path.join(src_path, f)
            dest_file_path = os.path.join(dest_path, f)
            os.system(f"cp -r {src_file_path} {dest_path}")
        gen_train_cmd = "python gen_yolo_txt.py --folder_path {}".format(dest_path.replace("labels", "images"))
        os.system(gen_train_cmd)

val_13cls = ["synthetic_only", "synthetic_full"]
for folder in val_13cls:
    src_path = os.path.join(root, folder, "labels/val")
    dest_path = os.path.join(root, folder, "labels_13cls/val")
    os.makedirs(dest_path, exist_ok=True)
    for f in os.listdir(src_path):
        src_file_path = os.path.join(src_path, f)
        dest_file_path = os.path.join(dest_path, f)
        os.system(f"cp -r {src_file_path} {dest_path}")


print("Done copying training data to different settings.")

copy_names = ["synthetic_only"] + copy_names
project_name = root.split("/")[-1]
sample_cmd = "python -m torch.distributed.launch --nproc_per_node 2 --master_port 1888 train.py --device 0,1  --epochs 200 --batch-size 32 --weights yolov7  --project 0919/{}/{}/ --cfg cfg/training/yolov7-8cls.yaml --data {}/{}/data.yaml"

with open("dsec_{}.txt".format(project_name), "w") as f:
    for name in copy_names:
        cmd = sample_cmd.format(project_name, name, project_name, name)
        f.write(cmd + "\n")
