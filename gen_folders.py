import os

src_folder = "/home/bhzhang/Documents/code/EfficientMultiModal/data/caltech-101/101_ObjectCategories"
optical_flow_root = "/home/bhzhang/Documents/tools/RAFT/results/NCL_caltech-101_former"
dest_folder = "/home/bhzhang/Documents/code/EfficientMultiModal/data/generated_caltech-101_opticalflow_former"

cmd_tpl = 'python gen_output_opticalflow.py --of_norm_factor 50 --repeat 4 --of_scales 0.2 0.5 1.0 1.5 2.0 2.5 3.0 5.0 --flow_max 200 --controlnet_path exp/RAFTflow_norm50 --pretrained_model_path "stable-diffusion-v1-5/stable-diffusion-v1-5" --rgb_input_dir {} --output_dir {} --optical_flow_input_dir {} --save_result_only'

for category in os.listdir(src_folder):
    category_path = os.path.join(src_folder, category)
    optical_flow_folder = os.path.join(optical_flow_root, category, "flow")
    if os.path.isdir(category_path):
        dest_category_path = os.path.join(dest_folder, category)
        os.makedirs(dest_category_path, exist_ok=True)

        # for img_file in os.listdir(category_path):
        cmd = cmd_tpl.format(
            category_path,
            dest_category_path,
            optical_flow_folder
        )
        print(cmd)
        os.system(cmd)





