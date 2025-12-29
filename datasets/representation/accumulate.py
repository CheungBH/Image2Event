import numpy as np

def accumulate_0(image, x, y, p, reverse=False):

    for x_, y_, p_ in zip(x, y, p):
        if p_ == 0:
            image[y_, x_] = np.array([0, 0, 255])
        else:
            image[y_, x_] = np.array([255, 0, 0])
    return image

def accumulate(frame_events, sensor_size, frame_buffer, reverse=False):
    sensor_height = sensor_size[1]
    if reverse:
        pos_mask = frame_events['p'] == 0
        neg_mask = frame_events['p'] == 1
    else:
        pos_mask = frame_events['p'] == 1
        neg_mask = frame_events['p'] == 0
    # np.add.at(frame_buffer[:, :, 0],  # 红色通道
    #           (frame_events['y'][pos_mask], frame_events['x'][pos_mask]), 255)
    # np.add.at(frame_buffer[:, :, 2],  # 蓝色通道
    #           (frame_events['y'][neg_mask], frame_events['x'][neg_mask]), 255)
    # 关键修正：翻转Y轴坐标
    y_pos = sensor_height - 1 - frame_events['y'][pos_mask]  # 正事件的Y坐标
    y_neg = sensor_height - 1 - frame_events['y'][neg_mask]  # 负事件的Y坐标

    # 红色通道（正事件）
    np.add.at(frame_buffer[:, :, 0],
              (y_pos, frame_events['x'][pos_mask]), 255)

    # 蓝色通道（负事件）
    np.add.at(frame_buffer[:, :, 2],
              (y_neg, frame_events['x'][neg_mask]), 255)
    # 限制像素值范围
    frame_buffer = np.clip(frame_buffer, 0, 255)
    return frame_buffer
