import os
import shutil

def organize_direction_dataset(root_folder, dest_folder):
    direction_kw = ["forward", "reverse"]
    src_folders = os.listdir(root_folder)
    src_folder_unique = ["-".join(f.split("-")[:-1]) for f in src_folders if os.path.isdir(os.path.join(root_folder, f))]
    src_folder_unique = list(set(src_folder_unique))
    for folder in src_folder_unique:
        for direction in direction_kw:
            dest_dir = os.path.join(dest_folder, "{}".format(folder))
            os.makedirs(dest_dir, exist_ok=True)
            direction_folder = f"{folder}-{direction}"
            direction_folder_path = os.path.join(root_folder, direction_folder)
            event_image_file = os.path.join(direction_folder_path, "Accumulate.png")
            dest_image_file = os.path.join(dest_dir, "Accumulate-{}.png".format(direction))
            shutil.copy(event_image_file, dest_image_file)
            if direction == "forward":
                flow_path = os.path.join(direction_folder_path, "flow_up.npy")
                dest_flow_path = os.path.join(dest_dir, "flow_up.npy")
                shutil.copy(flow_path, dest_flow_path)
                raw_image_file = os.path.join(direction_folder_path, "raw.png")
                dest_raw_image_file = os.path.join(dest_dir, "raw.png")
                shutil.copy(raw_image_file, dest_raw_image_file)



if __name__ == '__main__':
    root_folder = "/home/bhzhang/Documents/code/EventDiffusion/data/RAFT_flow_original/train"
    final_folder = "/home/bhzhang/Documents/code/EventDiffusion/data/RAFT_flow_original/train_direction"
    # os.makedirs(dest_folder, exist_ok=True)

    for file_name in os.listdir(root_folder):
        src_folder = os.path.join(root_folder, file_name)
        dest_folder = os.path.join(final_folder, file_name)
        organize_direction_dataset(src_folder, dest_folder)