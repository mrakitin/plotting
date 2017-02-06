import os

import matplotlib.pyplot as plt
import numpy as np
import uti_io
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable


PIXEL_SIZE = 0.06  # mm


def plot_slices(image_file, show_channel_image=False, show_sliced_image=True, channel='red', num_steps=10):
    imobj = uti_io.read_image(image_file)['raw_image']
    data = np.array(imobj)

    # Extract the red channel assuming it corresponds to maximum values (http://stackoverflow.com/a/12201744/4143531):
    channels = {
        'red': 0,
        'green': 1,
        'blue': 2,
    }
    selected_channel = data[..., channels[channel]]
    from scipy.ndimage import zoom
    zoom_coef = 50
    selected_channel = zoom(selected_channel, zoom_coef)

    # Convert units from pixels to mm:
    x_range = np.linspace(0, selected_channel.shape[0] * PIXEL_SIZE, selected_channel.shape[0])
    y_range = np.linspace(0, selected_channel.shape[1] * PIXEL_SIZE, selected_channel.shape[1])

    x_ticks = np.linspace(0, selected_channel.shape[0] * PIXEL_SIZE/zoom_coef, num_steps)
    y_ticks = np.linspace(0, selected_channel.shape[1] * PIXEL_SIZE/zoom_coef, num_steps)

    # Organize a grid for final image:
    fig = plt.figure(1, figsize=(8, 8))
    grid = ImageGrid(
        fig, 111,
        nrows_ncols=(2, 2),
        axes_pad=0.2,
    )

    # Show the image with the selected channel:
    ax = grid[2]
    ax.imshow(
        selected_channel,
        cmap='gray',
        extent=[
            y_range[0], y_range[-1],
            x_range[0], x_range[-1],
        ]
    )

    # Select the slices and plot them:
    middle_x = selected_channel[:, int(selected_channel.shape[1] / 2)]
    middle_y = selected_channel[int(selected_channel.shape[0] / 2), :]

    # Plot slices:
    axes_dict = {
        'x': 'vertical slice',
        'y': 'horizontal slice',
    }
    for i, k in enumerate(axes_dict.keys()):
        d = locals()['middle_{}'.format(k)]
        if k == 'x':
            ax = grid[3]
            ax.plot(d[::-1], locals()['{}_range'.format(k)])
        elif k == 'y':
            ax = grid[0]
            ax.plot(locals()['{}_range'.format(k)], d)
            # grid[0].set_xticks(x_ticks)
        ax.grid()
        parms = [channel, axes_dict[k], len(d)]
        title = '{} channel / {} in center / {} points'.format(*parms)
        fname = '{}_channel_{}_{}_points'.format(*parms)
        plt.title(title)
        plt.xlim(locals()['{}_range'.format(k)][0], locals()['{}_range'.format(k)][-1])
        # plt.xticks(locals()['{}_ticks'.format(k)])
        plt.xlabel('Coordinate [mm]')
        plt.ylabel('Intensity [arb.units]')
        # plt.grid()
        # if show_sliced_image:
        #     plt.show()
        # else:
        #     plt.savefig(os.path.join(imdir, '{}.tif'.format(fname)))
        # _clear_plots(plt)

    # This affects all axes as share_all = True.
    grid.axes_llc.set_yticklabels(x_ticks)
    grid.axes_llc.set_xticklabels(y_ticks)

    # plt.tight_layout()
    if show_channel_image:
        plt.show()
    else:
        plt.savefig(os.path.join(imdir, '{}_channel.tif'.format(channel)))
    _clear_plots(plt)

    return


def _clear_plots(plot):
    plot.close()
    plot.clf()
    plot.cla()


if __name__ == '__main__':
    imdir = 'C:\\Users\\Maksim\\Documents\\Work\\Beamlines\\ESM\\2017-02-02 ESM Diagon simulations\\ESM_images'
    # imfile = 'exp_22.5mm.tif'
    # imfile = 'exp_22.5mm_not_square.tif'
    # imfile = 'exp_22.5mm_narrow.tif'
    imfile = 'exp_22.5mm_narrow.png'
    impath = os.path.abspath(os.path.join(imdir, imfile))
    show_channel_image = True
    show_sliced_image = False
    channel = 'red'

    plot_slices(
        image_file=impath,
        show_channel_image=show_channel_image,
        show_sliced_image=show_sliced_image,
        channel=channel,
    )
