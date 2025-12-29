import os

label_folder = "/home/bhzhang/Documents/datasets/DSEC_dataset_1004/opticalflow_full/synthetic_only/labels/train"
labels = os.listdir(label_folder)

for label in labels:
    label_path = os.path.join(label_folder, label)
    with open(label_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if int(line.split(" ")[0]) > 2:
                print("OKOKOKOKO")
print("Done checking labels.")