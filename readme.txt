


Download KITTI dataset


train flow
python train_flow.py --dataset kitti --restore_ckpt RAFT/raft-kitti.pth --gpus 0 --num_steps 50000 --batch_size 5 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85 --mixed_precision --out_dir checkpoints/flow_kitti --dataset KITTI --dataset_root /home/bhzhang/Documents/tools/RAFT/dataset/KITTI
python train_flow.py --dataset DSEC_RAFT --restore_ckpt checkpoints/flow_kitti/final.pth --gpus 0 --num_steps 50000 --batch_size 8 --lr 0.0001 --image_size 480 640 --wdecay 0.00001 --gamma=0.85 --mixed_precision --out_dir checkpoints/flow_dsec_raft --dataset DSEC_RAFT --dataset_root /home/bhzhang/Documents/code/EventDiffusion/data/RAFT_flow_dataset

eval flow
python evaluate_flow.py --model checkpoints/flow_kitti/049999.pth --dataset KITTI --dataset_root /home/bhzhang/Documents/tools/RAFT/dataset/KITTI --phase training
python evaluate_flow.py --model checkpoints/flow_dsec_raft/final.pth --dataset DSEC_RAFT --dataset_root /home/bhzhang/Documents/code/EventDiffusion/data/RAFT_flow_dataset --phase test

inference flow
python inference_flow.py --output_folder assets/DSEC_RAFT_single --input_folder /home/bhzhang/Documents/code/EventDiffusion/data/RAFT_flow_dataset_simple/test/image --model checkpoints/flow_dsec_raft/final.pth



Download DSEC_det

cd datasets
python dsec_preprocess.py
python dsec_gen_dataset.py

cd ..
python train_controlnet.py --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" --output_dir=exp/1227_mse --train_data_dir=data/RAFT_flow_dataset/train  --validation_data_dir=data/RAFT_flow_dataset/test  --resolution=512 --learning_rate=1e-5 --train_batch_size=6 --num_train_epochs=30 --add_optical_flow --of_norm_factor 50 --checkpointing_steps 1000
python inference_controlnet.py --controlnet_path exp/1227_mse --pretrained_model_path "stable-diffusion-v1-5/stable-diffusion-v1-5" --rgb_input_dir /home/bhzhang/Documents/code/EventDiffusion/data/RAFT_flow_dataset_simple/test/image --output_dir assets/DSEC_RAFT_event --optical_flow_input_dir /home/bhzhang/Documents/code/EventDiffusion/data/RAFT_flow_dataset_simple/test/optical_flow --of_norm_factor 50 --of_scales 1.0 --flow_max -1 --save_result_only --binarize
python evaluate_controlnet.py /home/bhzhang/Documents/code/Image2Event/assets/DSEC_RAFT_event /home/bhzhang/Documents/code/EventDiffusion/data/RAFT_flow_dataset_simple/test/image

python cross_domain_generation.py --source_flow_stats_dir /home/bhzhang/Documents/code/EventDiffusion/data/RAFT_flow_dataset/train/optical_flow --controlnet_path exp/1227_mse --pretrained_model_path "stable-diffusion-v1-5/stable-diffusion-v1-5" --rgb_input_dir /home/bhzhang/Documents/code/EventDiffusion/data/RAFT_flow_dataset_simple/test/image --output_dir assets/tmp --optical_flow_input_dir /home/bhzhang/Documents/code/EventDiffusion/data/RAFT_flow_dataset_simple/test/optical_flow --of_norm_factor 50 --of_scales 1.0 2.0 3.0 --binarize

