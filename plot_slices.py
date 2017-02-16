import os

import matplotlib.pyplot as plt
import numpy as np
import uti_io
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import zoom

from load_hdf5 import load_hdf5, get_gaps_range, gap_to_num, cut_data


def get_data_from_image(image_file, channel):
    image_dir = os.path.dirname(image_file)
    image_name = os.path.basename(image_file)

    img = uti_io.read_image(image_file, ignore_bottom_limit=True)['raw_image']
    data_rgb = np.array(img)
    img = img.convert('HSV')
    data_hsv = np.array(img)

    # Extract the red channel assuming it corresponds to maximum values (http://stackoverflow.com/a/12201744/4143531):
    channels = {
        'red': 0,
        'green': 1,
        'blue': 2,
    }
    if channel == 'all' or channel is None:
        # Brightness/value (https://en.wikipedia.org/wiki/HSL_and_HSV, http://stackoverflow.com/a/15794784/4143531):
        data = data_hsv[:, :, 2]
    else:
        data = data_rgb[..., channels[channel]]

    return {
        'data': data,
        'image_dir': image_dir,
        'image_name': image_name,
    }


def plot_slices(data, image_dir, image_name, show_channel_image=False, channel=None, num_steps=10, zoom_coef=1,
                pixel_size=0.06, cmap='gray'):
    """Plot selected channel of the 2d image and the corresponding horizontal and vertical central cuts:

    ...
    :param pixel_size: [mm]
    :return: None
    """

    if zoom_coef > 1:
        data = zoom(data, zoom_coef)

    # Convert units from pixels to mm:
    if type(pixel_size) in [list, tuple]:
        pixel_size_x = pixel_size[0]
        pixel_size_y = pixel_size[1]
    else:
        pixel_size_x = pixel_size
        pixel_size_y = pixel_size
    x_range = np.linspace(0, data.shape[1] * pixel_size_x, data.shape[1])
    y_range = np.linspace(0, data.shape[0] * pixel_size_y, data.shape[0])

    resize = int(np.ceil(data.shape[1] / data.shape[0] * num_steps))
    x_ticks = np.linspace(0, data.shape[0] * pixel_size_y / zoom_coef, num_steps)
    y_ticks = np.linspace(0, data.shape[1] * pixel_size_x / zoom_coef, resize)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Show the image with the selected channel:
    ax.imshow(
        data,
        cmap=cmap,
        extent=[
            y_ticks[0], y_ticks[-1],
            x_ticks[0], x_ticks[-1],
        ]
    )
    ax.set_xticks(y_ticks)
    ax.set_yticks(x_ticks)

    # Select the slices and plot them:
    horiz_center = int(data.shape[1] / 2)
    vert_center = int(data.shape[0] / 2)

    vert_center_cut = data[:, horiz_center]
    horiz_center_cut = data[vert_center, :]

    # Set horizontal and vertical cut lines:
    line_color = 'yellow' if (channel == 'all' or channel is None) else channel
    ax.axvline(x=x_range[horiz_center] / zoom_coef, color=line_color)
    ax.axhline(y=y_range[vert_center] / zoom_coef, color=line_color)

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
        infile_name, infile_ext = os.path.splitext(image_name)
        ch = channel if channel else ''
        outfile_name = '{}_cuts_{}_channel{}'.format(infile_name, ch, infile_ext)
        plt.savefig(os.path.join(image_dir, outfile_name))

    _clear_plots(plt)

    return


def _clear_plots(plot):
    plot.close()
    plot.clf()
    plot.cla()


if __name__ == '__main__':
    # input_type = 'hdf5'
    # input_type = 'image'
    input_type = 'dat_file'

    # show_channel_image = True
    show_channel_image = False

    # channels = ['red', 'green', 'blue']
    # channels = ['all']
    # channels = ['red']
    channels = [None]

    gap = 25.5  # mm

    for channel in channels:
        if input_type == 'image':
            imdir = 'C:\\Users\\Maksim\\Documents\\Work\\Beamlines\\ESM\\2017-02-02 ESM Diagon simulations\\ESM_images'
            pixel_size = 0.06  # mm
            imfile = 'exp_22.5mm_not_square.tif'
            # imfile = 'exp_22.5mm.tif'
            # imfile = 'exp_22.5mm_narrow.tif'
            # imfile = 'exp_22.5mm_narrow.png'

            # pixel_size = 12 / 276.  # 0.043 mm
            # imfile = 'calc_230eV_22.5mm.png'

            pixel_size = 12 / 800.
            impath = 'image_g_{}.png'.format(gap)

            # imfile = 'g_{}.png'.format(gap)
            # impath = os.path.abspath(os.path.join(imdir, imfile))
            d = get_data_from_image(image_file=impath, channel=channel)
            data = d['data']
            image_dir = d['image_dir']
            image_name = d['image_name']
        elif input_type == 'hdf5':
            # pixel_size = 0.035  # mm
            pixel_size = (0.034, 0.035)  # mm

            data_dir = 'hdf5_to_images'
            hdf5_file = '107fe4cc-5e40-4922-acb4_000000.h5'
            d = load_hdf5(data_dir=data_dir, hdf5_file=hdf5_file)
            data = d['data']
            num_of_images = d['num_of_images']
            gaps_list = get_gaps_range(num_of_images, start=16, step=0.5)
            data = cut_data(data, gap_to_num(gap, gaps_list=gaps_list))
            image_dir = data_dir
            image_name = 'hdf5_g_{}.png'.format(gap)
        elif input_type == 'dat_file':
            from plotting import prepare_data
            energy=300
            dat_file = 'c:\\Users\\Maksim\\Documents\\Work\\Beamlines\\ESM\\2017-02-02 ESM Diagon simulations\\dat_files\\e_{}eV_vemit_30e-12_29.462m\\g_{}.dat'.format(
                energy,
                gap)
            data, x_values, y_values = prepare_data(dat_file)
            image_dir = ''
            image_name = 'dat_g_{}_e_{}.png'.format(gap, energy)
            pixel_size = (
                (x_values.max() - x_values.min()) * 1000 / len(x_values),  # mm
                (y_values.max() - y_values.min()) * 1000 / len(y_values),  # mm
            )
        else:
            raise ValueError('Input type <{}> is not allowed.'.format(input_type))

        plot_slices(
            data=data,
            image_dir=image_dir,
            image_name=image_name,
            show_channel_image=show_channel_image,
            channel=channel,
            pixel_size=pixel_size,
        )
