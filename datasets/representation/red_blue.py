
# EventGPT using

import numpy as np

def red_blue_display(frame_events, sensor_size, flip_y=True, **kwargs):
    x, y, p = frame_events['x'], frame_events['y'], frame_events['p']
    height, width = sensor_size[1], sensor_size[0]
    event_image = np.ones((height, width, 3), dtype=np.uint8) * 255

    for x_, y_, p_ in zip(x, y, p):
        if flip_y:
            y_ = height - 1 - y_
        if p_ == 0:
            event_image[y_, x_] = np.array([0, 0, 255])  # Blue for negative polarity
        else:
            event_image[y_, x_] = np.array([255, 0, 0])  # Red for positive polarity

    return event_image