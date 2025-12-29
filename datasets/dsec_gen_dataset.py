import os
import json

def create_folders(root_folder, out_events_folder, out_rgb_folder, out_flow_folder, event_targets=["Accumulate.png"]):
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        for sample in os.listdir(folder_path):

            rgb_image_path = os.path.join(folder_path, sample, "raw.png")
            out_rgb_image_path = os.path.join(out_rgb_folder, "{}-{}.jpg".format(folder, sample))
            os.system(f"cp {rgb_image_path} {out_rgb_image_path}")
            optical_flow_path = os.path.join(folder_path, sample, "flow.npy")
            out_optical_flow_path = os.path.join(out_flow_folder, "{}-{}-flow.npy".format(folder, sample))
            os.system(f"cp {optical_flow_path} {out_optical_flow_path}")
            for event_target in event_targets:
                event_image_path = os.path.join(folder_path, sample, event_target)
                out_event_image_path = os.path.join(out_events_folder,
                                                    "{}-{}-{}".format(folder, sample, event_target))
                os.system(f"cp {event_image_path} {out_event_image_path}")


def generate_metadata(base_dir, event_metas):

    event_names = event_metas[0]
    images_dir = os.path.join(base_dir, "images")
    cond_dir = os.path.join(base_dir, "conditioning_images")
    optical_flows_dir = os.path.join(base_dir, "optical_flow")
    prompt_tpl = "convert to event frame using going {} method"
    image_files, cond_files = [], []
    metadata = []

    for file in os.listdir(cond_dir):
        # if file.endswith('.png') or file.endswith('.jpg'):
        for event_name in event_names:
            cond_files.append(os.path.join(cond_dir, file))
            base_name = os.path.splitext(file)[0]
            image_file = os.path.join(images_dir, f"{base_name}-{event_name}")
            image_files.append(image_file)
            prompt_kw = "reverse" if "reverse" in base_name.lower() else "forward"
            prompt = prompt_tpl.format(prompt_kw)
            metadata.append({
                "text": prompt,
                "image": "/".join(image_file.split("/")[-2:]),
                "conditioning_image": "/".join(cond_files[-1].split("/")[-2:]),
                "optical_flow": os.path.join(optical_flows_dir.split("/")[-1], f"{base_name}-flow.npy").replace("\\", "/"),
            })


    output_path = os.path.join(base_dir, "metadata.jsonl")
    with open(output_path, "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")



if __name__ == '__main__':
    phases = ["train", "test"]
    root_folder_src = "data/DSEC_dataset"
    output_folder_src = "data/RAFT_flow_dataset"
    event_targets = ["event.png"]

    def remove_all_DS_Store(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.startswith('.'):
                    os.remove(os.path.join(root, file))
    for phase in phases:
        root_folder = os.path.join(root_folder_src, phase)
        output_folder = os.path.join(output_folder_src, phase)
        remove_all_DS_Store(root_folder)
        out_events_folder = os.path.join(output_folder, "images")
        out_rgb_folder = os.path.join(output_folder, "conditioning_images")
        out_flow_folder = os.path.join(output_folder, "optical_flow")
        os.makedirs(out_rgb_folder, exist_ok=True)
        os.makedirs(out_events_folder, exist_ok=True)
        os.makedirs(out_flow_folder, exist_ok=True)
        create_folders(root_folder, out_events_folder, out_rgb_folder, out_flow_folder, event_targets=event_targets)
        generate_metadata(output_folder, event_metas=[event_targets])
        os.system("cp dataset_script.py {}".format(output_folder))
