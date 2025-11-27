import numpy as np
from matplotlib import pyplot as plt


def imshow(inp, title=None, filename=None):
    """Display image for Tensor."""

    # change order of channels
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    # plt.imshow(inp)
    fig, ax = plt.subplots(figsize=(15, 9), dpi=200)  # Adjust figsize and dpi as needed
    ax.imshow(inp)
    if title is not None:
        plt.title(title)

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

    plt.pause(0.001)  # pause a bit so that plots are updated

