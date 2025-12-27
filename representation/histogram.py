import numpy as np

# EventClip using

def make_event_histogram(x, y, p, red, blue, shape, thresh=10., count_non_zero=False, background_mask=True, flip_y=True):
    """Event polarity    histogram."""
    # count the number of positive and negative events per pixel
    W, H, _ = shape
    if flip_y:
        y = H - 1 - y
    pos_x, pos_y = x[p > 0].astype(np.int32), y[p > 0].astype(np.int32)
    pos_count = np.bincount(pos_x + pos_y * W, minlength=H * W).reshape(H, W)
    neg_x, neg_y = x[p < 0].astype(np.int32), y[p < 0].astype(np.int32)
    neg_count = np.bincount(neg_x + neg_y * W, minlength=H * W).reshape(H, W)
    hist = np.stack([pos_count, neg_count], axis=-1)  # [H, W, 2]

    # remove hotpixels, i.e. pixels with event num > thresh * std + mean
    if thresh > 0:
        if not count_non_zero:
            mean = hist[hist > 0].mean()
            std = hist[hist > 0].std()
        else:
            mean = hist.mean()
            std = hist.std()
        hist[hist > thresh * std + mean] = 0

    # normalize
    hist = hist.astype(np.float32) / hist.max()  # [H, W, 2]

    # colorize
    cmap = np.stack([red, blue], axis=0).astype(np.float32)  # [2, 3]
    img = hist @ cmap  # [H, W, 3]

    # alpha-masking with pure white background
    if background_mask:
        weights = np.clip(hist.sum(-1, keepdims=True), a_min=0, a_max=1)
        background = np.ones_like(img) * 255.
        img = img * weights + background * (1. - weights)

    img = np.round(img).astype(np.uint8)  # [H, W, 3], np.uint8 in (0, 255)

    return img

def to_event_histogram(event_npy, sensor_size, red=[255, 0, 0], blue=[0, 0, 255], **kwargs):
    """Convert events to event histogram."""
    x, y, p = event_npy['x'], event_npy['y'], event_npy['p']
    x, y, p = x.astype(np.int32), y.astype(np.int32), p.astype(np.int32)
    hist_img = make_event_histogram(x, y, p, red=red, blue=blue,
                                    shape=sensor_size)
    return hist_img