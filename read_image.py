import os

import matplotlib.pyplot as plt
import numpy as np
import uti_io
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import zoom


def plot_slices(image_file, show_channel_image=False, channel='red', num_steps=10, zoom_coef=1, pixel_size=0.06):
    """Plot selected channel of the 2d image and the corresponding horizontal and vertical central cuts:

    :param image_file:
    :param show_channel_image:
    :param channel:
    :param num_steps:
    :param zoom_coef:
    :param pixel_size: [mm]
    :return: None
    """
    image_dir = os.path.dirname(image_file)

    img = uti_io.read_image(image_file, ignore_bottom_limit=True)['raw_image']
    data = np.array(img)

    # Extract the red channel assuming it corresponds to maximum values (http://stackoverflow.com/a/12201744/4143531):
    channels = {
        'red': 0,
        'green': 1,
        'blue': 2,
    }
    selected_channel = data[..., channels[channel]]

    if zoom_coef > 1:
        selected_channel = zoom(selected_channel, zoom_coef)

    # Convert units from pixels to mm:
    x_range = np.linspace(0, selected_channel.shape[1] * pixel_size, selected_channel.shape[1])
    y_range = np.linspace(0, selected_channel.shape[0] * pixel_size, selected_channel.shape[0])

    resize = int(np.ceil(selected_channel.shape[1] / selected_channel.shape[0] * num_steps))
    x_ticks = np.linspace(0, selected_channel.shape[0] * pixel_size / zoom_coef, num_steps)
    y_ticks = np.linspace(0, selected_channel.shape[1] * pixel_size / zoom_coef, resize)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Show the image with the selected channel:
    ax.imshow(
        selected_channel,
        cmap='gray',
        extent=[
            y_ticks[0], y_ticks[-1],
            x_ticks[0], x_ticks[-1],
        ]
    )
    ax.set_xticks(y_ticks)
    ax.set_yticks(x_ticks)

    # Select the slices and plot them:
    horiz_center = int(selected_channel.shape[1] / 2)
    vert_center = int(selected_channel.shape[0] / 2)

    vert_center_cut = selected_channel[:, horiz_center]
    horiz_center_cut = selected_channel[vert_center, :]

    # Set horizontal and vertical cut lines:
    ax.axvline(x=x_range[horiz_center] / zoom_coef, color=channel)
    ax.axhline(y=y_range[vert_center] / zoom_coef, color=channel)

    ax.set_xlabel('Coordinate [mm]')
    ax.set_ylabel('Coordinate [mm]')

    # Plot the cuts:
    divider = make_axes_locatable(ax)

    ax_horiz_cut = divider.append_axes('top', 1.5, pad=0.3, sharex=ax)
    ax_vert_cut = divider.append_axes('right', 1.5, pad=0.6, sharey=ax)

    ax_horiz_cut.plot(x_range / zoom_coef, horiz_center_cut)
    ax_horiz_cut.grid()
    ax_horiz_cut.set_ylabel('Intensity [arb.units]')

    ax_vert_cut.plot(vert_center_cut[::-1], y_range / zoom_coef)
    ax_vert_cut.grid()
    ax_vert_cut.set_xlabel('Intensity [arb.units]')

    plt.tight_layout()

    if show_channel_image:
        plt.show()
    else:
        infile_name, infile_ext = os.path.splitext(os.path.basename(image_file))
        outfile_name = '{}_cuts_{}_channel{}'.format(infile_name, channel, infile_ext)
        plt.savefig(os.path.join(image_dir, outfile_name))

    _clear_plots(plt)

    return


def _clear_plots(plot):
    plot.close()
    plot.clf()
    plot.cla()


if __name__ == '__main__':
    imdir = 'C:\\Users\\Maksim\\Documents\\Work\\Beamlines\\ESM\\2017-02-02 ESM Diagon simulations\\ESM_images'
    # pixel_size = 0.06  # mm
    # imfile = 'exp_22.5mm_not_square.tif'
    # imfile = 'exp_22.5mm.tif'
    # imfile = 'exp_22.5mm_narrow.tif'
    # imfile = 'exp_22.5mm_narrow.png'

    pixel_size = 12 / 276.  # 0.043 mm
    imfile = 'calc_230eV_22.5mm.png'

    impath = os.path.abspath(os.path.join(imdir, imfile))
    # show_channel_image = True
    show_channel_image = False

    channels = ['red', 'green', 'blue']
    for channel in channels:
        plot_slices(
            image_file=impath,
            show_channel_image=show_channel_image,
            channel=channel,
            pixel_size=pixel_size,
        )
