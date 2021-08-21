import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid

def plot_image_grid(y, title=None, display=True, save_path=None, figsize=(10, 10)):
    """Plot and optionally save an image grid with matplotlib"""
    fig = plt.figure(figsize=figsize)
    num_rows = int(np.floor(np.sqrt(y.shape[0])))
    grid = ImageGrid(fig, 111, nrows_ncols=(num_rows, num_rows), axes_pad=0.1)
    for ax in grid: 
        ax.set_axis_off()
    for ax, im in zip(grid, y):
        ax.imshow(im)
    fig.suptitle(title, fontsize=18)
    fig.subplots_adjust(top=0.98)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if display:
        plt.show()
    else:
        plt.close()