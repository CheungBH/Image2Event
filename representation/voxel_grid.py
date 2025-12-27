
# EventSAM using

import numpy as np
from .voxel_grid_utils.voxel_grid import events_to_voxel


def voxel_grid_representation(events, sensor_size, flip_y=True, **kwargs):
    xs, ys, ts, ps = events["x"], events["y"], events["t"], events["p"]
    if flip_y:
        ys = sensor_size[1] - 1 - ys
    voxel = events_to_voxel(xs, ys, ts, ps, B=3, sensor_size=sensor_size, temporal_bilinear=True)  # voxel: [3,H,W]
    voxel = voxel.transpose((2, 1, 0))

    voxel = voxel - np.min(voxel)
    voxel_img = 255 * (voxel / np.max(voxel))
    return voxel_img.astype(np.uint8)


