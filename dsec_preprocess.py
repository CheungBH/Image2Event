import shutil
from PIL import Image
from RAFT.raft import RAFT
from RAFT.utils.utils import InputPadder
import torch
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from typing import Dict
import hdf5plugin
import argparse
from representation import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', help="restore checkpoint")
parser.add_argument('--path', help="dataset for evaluation")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
args = parser.parse_args()

RAFT_path = "RAFT/raft-kitti.pth"
model = torch.nn.DataParallel(RAFT(args))
model.load_state_dict(torch.load(RAFT_path))

model = model.module
model.to("cuda:0")
model.eval()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def RAFT_load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to("cuda:0")


def load_timestamp(file_path):
    """Load a timestamp file and return a list of timestamps."""
    with open(file_path, "r") as file:
        lines = file.readlines()[1:]
        lines = [line[:-1].split(",") for line in lines]
    first_timestamp = float(lines[0][0])

    new_lines = []
    for line in lines:
        new_lines.append(float(line[0]) - first_timestamp)
        # new_lines.append(float(line[1]) - first_timestamp)

    return new_lines


class EventVisualizer:
    def __init__(self, sensor_size=(346, 260, 2)):
        self.sensor_size = sensor_size  # (width, height)
        self._sanity_check()

    def _sanity_check(self):

        assert self.sensor_size[0] > 0
        assert self.sensor_size[1] > 0
        assert self.sensor_size[2] in (1, 2)

    def generate_frame_sequence(self,
                                events_file,
                                total_events,
                                sensor_size,
                                frame_interval,  # 单位：秒
                                output_dir,
                                root_dir,
                                sample_interval=1,
                                temporal_intervals=[1],
                                reverse=False,
                                chunk_size=1000000):

        sensor_size = (sensor_size[0], sensor_size[1], 2)
        os.makedirs(output_dir, exist_ok=True)
        img_dir = os.path.join(folder_path, "images", direction, "distorted")


        time_bins = frame_interval
        direction_kw = "reverse" if reverse else "forward"

        pbar = tqdm(total=len(time_bins), desc="Generating event frames", unit="frame")
        events_group = events_file['events']

        frame_RGB_interval = 50000
        for img_idx, frame_id in enumerate(range(len(time_bins))):
            pbar.update(1)
            if img_idx % sample_interval != 0:
                continue

            for temporal_interval in temporal_intervals:
                if frame_id + temporal_interval >= len(time_bins):
                    continue
                output_dir_path = os.path.join(output_dir,
                                               "{}-{}-{}".format(str(img_idx), temporal_interval, direction_kw))
                if os.path.exists(output_dir_path):
                    continue

                start_time = time_bins[frame_id]
                end_time = frame_RGB_interval * temporal_interval + start_time

                # Find event indices for the current time window
                t_arr = events_group['t']
                start_event_idx = np.searchsorted(t_arr, start_time, side='left')
                end_event_idx = np.searchsorted(t_arr, end_time, side='right')

                if start_event_idx >= end_event_idx:
                    continue

                # Load only the required chunk of events
                p = events_group['p'][start_event_idx:end_event_idx].astype(bool)
                if reverse:
                    p = ~p

                frame_events = np.core.records.fromarrays(
                    [
                        events_group['t'][start_event_idx:end_event_idx],
                        events_group['x'][start_event_idx:end_event_idx],
                        events_group['y'][start_event_idx:end_event_idx],
                        p
                    ],
                    dtype=[('t', np.uint64), ('x', np.uint16), ('y', np.uint16), ('p', np.bool_)]
                )
                frame_events = self.filter_coordinates(frame_events)

                img_file = os.path.join(img_dir, f"{frame_id:06d}.png")
                flow_img_file = os.path.join(img_dir, f"{frame_id + temporal_interval:06d}.png")
                if not os.path.exists(img_file):
                    continue

                image1, image2 = RAFT_load_image(img_file), RAFT_load_image(flow_img_file)


                os.makedirs(output_dir_path, exist_ok=True)
                shutil.copy(img_file, os.path.join(output_dir_path, "raw.png"))

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                with torch.no_grad():
                    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

                flow_up_unpad = padder.unpad(flow_up)
                np.save(os.path.join(output_dir_path, "flow.npy"), flow_up_unpad.detach().cpu().numpy())

                frame_buffer = np.zeros((self.sensor_size[1], self.sensor_size[0], 3), dtype=np.uint8)
                frame_buffer = accumulate(frame_events, sensor_size=self.sensor_size, frame_buffer=frame_buffer)
                output_path = os.path.join(output_dir_path, "event.png")
                plt.imsave(output_path, frame_buffer, origin='lower')
                frame_buffer.fill(0)

        pbar.close()
        logger.info(f"Finished processing for {output_dir}")

    def load_events_safely(self, h5_path: str, shuffle_p=False) -> (h5py.File, int):
        """
        Safely opens an HDF5 event file and returns the file handle and total event count.
        The file must be closed by the caller.
        """
        events_file = h5py.File(h5_path, 'r')
        total_events = events_file['events']['t'].shape[0]

        # Basic validation
        timestamps = events_file['events']['t']
        if np.median(timestamps[:10000]) < 100:
             logger.warning("Timestamp values seem low. Ensure they are in microseconds.")
        if not np.all(np.diff(timestamps) >= 0):
            logger.error(f"Timestamps in {h5_path} are not sorted. Processing may fail.")
            # raise ValueError("Timestamps must be sorted for efficient processing.")

        return events_file, total_events

    def filter_coordinates(self, events: np.ndarray):
        mask = (events['x'] < self.sensor_size[0]) & (events['y'] < self.sensor_size[1])
        return events[mask]

    def fix_timestamps(self, events: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if np.median(events['t']) < 100:  # 判断是否为无效时间
            logger.warning("检测到异常时间戳，正在生成线性时间序列...")
            interval = 1000  # 1ms间隔
            events['t'] = np.arange(len(events['t'])) * interval
        return events

    def generate_event_image(self, events: np.ndarray) -> np.ndarray:
        """生成安全的事件累积图像"""

        # 最终坐标验证
        assert events['x'].max() < self.sensor_size[0], \
            f"X坐标越界: {events['x'].max()} >= {self.sensor_size[0]}"
        assert events['y'].max() < self.sensor_size[1], \
            f"Y坐标越界: {events['y'].max()} >= {self.sensor_size[1]}"

        # 创建图像画布（height, width）
        image = np.zeros((self.sensor_size[1], self.sensor_size[0]), dtype=np.float32)

        # 高效累积操作
        np.add.at(image, (events['y'], events['x']), events['p'].astype(np.float32))

        # 归一化处理
        return (image - image.min()) / (image.max() - image.min() + 1e-6)


    def visualize(self, image: np.ndarray, output_path: str = None):
        """可视化图像（支持RGB）"""
        plt.figure(figsize=(12, 8))

        if image.ndim == 2:  # 灰度图
            plt.imshow(image, cmap='viridis', origin='upper', vmin=0, vmax=1)
        elif image.ndim == 3:  # RGB图
            plt.imshow(image.astype(np.uint8), origin='upper')

        # 中文字体配置
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        plt.colorbar(label='事件密度')
        plt.title("事件帧可视化")
        plt.xlabel("X坐标 (像素)")
        plt.ylabel("Y坐标 (像素)")

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"图像已保存至 {output_path}")
        else:
            plt.show()
        plt.close()


if __name__ == "__main__":
    SENSOR_SIZE = (640, 480)
    sample_interval = 5
    temporal_intervals = [1,2,3]
    reverse = [False, True]
    output_root = "data/DSEC_dataset/train"

    root_folder = "data/DSEC_det_original/train"
    kw = []

    original_folders = os.listdir(root_folder)
    print(f"original_folders: {original_folders}")
    if kw:
        folders = [folder for folder in original_folders if folder in kw]
    else:
        folders = original_folders
    print("Found {} folders matching keywords.".format(len(folders)))

    folders_path = [
        os.path.join(root_folder, folder) for folder in folders
    ]

    for folder_path in folders_path:
        # if len(kw) and not any(k in folder_path for k in kw):
        #     print("Skipping folder:", folder_path)
        #     continue

        print("Processing folder:", folder_path)
        directions = ["left"]
        folder_name = os.path.basename(folder_path)

        for direction in directions:
            event_path = os.path.join(folder_path, "events", direction, "events.h5")
            time_stamp_file = os.path.join(folder_path, "images", direction, "exposure_timestamps.txt")

            if not os.path.exists(event_path):
                continue

            output_path = os.path.join(output_root, "{}_{}".format(folder_name, direction))
            # if os.path.exists(output_path):
            #     print("Output path exists, skipping:", output_path)
            #     continue

            visualizer = EventVisualizer(sensor_size=(*SENSOR_SIZE, 2))
            frame_interval = load_timestamp(time_stamp_file)

            events_file, total_events = visualizer.load_events_safely(event_path, shuffle_p=False)

            try:
                for r in reverse:
                    visualizer.generate_frame_sequence(
                        events_file,
                        total_events,
                        SENSOR_SIZE,
                        frame_interval=frame_interval,
                        output_dir=output_path,
                        root_dir=folder_path,
                        sample_interval=sample_interval,
                        temporal_intervals=temporal_intervals,
                        reverse=r
                    )
            finally:
                events_file.close()  # Ensure the HDF5 file is always closed