


Download KITTI dataset



train flow
python train_flow.py --dataset kitti --restore_ckpt RAFT/raft-kitti.pth --gpus 0 --num_steps 50000 --batch_size 5 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85 --mixed_precision --out_dir checkpoints/flow_kitti --dataset KITTI --dataset_root /home/bhzhang/Documents/tools/RAFT/dataset/KITTI
python train_flow.py --dataset DSEC_RAFT --restore_ckpt checkpoints/flow_kitti/final.pth --gpus 0 --num_steps 50000 --batch_size 8 --lr 0.0001 --image_size 480 640 --wdecay 0.00001 --gamma=0.85 --mixed_precision --out_dir checkpoints/flow_dsec_raft --dataset DSEC_RAFT --dataset_root /home/bhzhang/Documents/code/EventDiffusion/data/RAFT_flow_dataset

python evaluate_flow.py --model checkpoints/flow/final.pth --dataset kitti
python evaluate_flow.py --model checkpoints/flow_dsec_raft/final.pth --dataset DSEC_RAFT



Download DSEC