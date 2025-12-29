import numpy as np
from typing import Tuple


def plot_animation(frames: np.ndarray, fig, save_path):
    """Helper function that animates a tensor of frames of shape (TCHW). If you run this in a
    Jupyter notebook, you can display the animation inline like shown in the example below.

    Parameters:
        frames: numpy array or tensor of shape (TCHW)

    Example:
        >>> import tonic
        >>> nmnist = tonic.datasets.NMNIST(save_to='./data', train=False)
        >>> events, label = nmnist[0]
        >>>
        >>> transform = tonic.transforms.ToFrame(
        >>>     sensor_size=nmnist.sensor_size,
        >>>     time_window=10000,
        >>> )
        >>>
        >>> frames = transform(events)
        >>> animation = tonic.utils.plot_animation(frames)
        >>>
        >>> # Display the animation inline in a Jupyter notebook
        >>> from IPython.display import HTML
        >>> HTML(animation.to_jshtml())

    Returns:
        The animation object. Store this in a variable to keep it from being garbage collected until displayed.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation
    except ImportError:
        raise ImportError(
            "Please install the matplotlib package to plot events. This is an optional"
            " dependency."
        )
    # fig = plt.figure(figsize=figsize)
    if frames.shape[1] == 2:
        rgb = np.zeros((frames.shape[0], 3, *frames.shape[2:]))
        rgb[:, 1:, ...] = frames
        frames = rgb
    if frames.shape[1] in [1, 2, 3]:
        frames = np.moveaxis(frames, 1, 3)
    plt.imshow(frames[0])
    plt.axis("off")
    fig.tight_layout()
    #
    # def animate(frame):
    #     ax.set_data(frame)
    #     return ax

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    # anim = animation.FuncAnimation(fig, animate, frames=frames, interval=100)
    # plt.show()
    # return anim
