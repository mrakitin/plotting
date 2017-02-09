import glob
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import uti_io
from mpl_toolkits.axes_grid1 import ImageGrid


def cut_pictures(image_dir, x_start, x_len, y_start, y_len, search_extention='tif', outdir=None, save=True, cut_every=3,
                 silent=False):
    image_list = glob.glob(os.path.join(image_dir, '*.{}'.format(search_extention)))
    cut_images = []
    for i, f in enumerate(image_list):
        if i % cut_every == 0:
            cut_data = read_single_pic(f, x_start, x_len, y_start, y_len, outdir, save=save)
            cut_images.append(cut_data)
            if not silent:
                print('File: {}'.format(f))
    return cut_images


def plot_grid(data_list, nrows, ncols, cmap='afmhot', save_dir='', first_set_only=True, show=False, grid_image=None,
              axes_pad=0.05, wider=3.5, want_plot_grid=True):
    num_in_set = nrows * ncols
    num_of_sets = int(np.ceil(len(data_list) / num_in_set))
    save_dir = os.path.abspath(save_dir)
    outname = os.path.basename(save_dir)

    min_value = np.min(data_list)
    max_value = np.max(data_list)

    if want_plot_grid:
        if not grid_image:
            grid_image = '{}_grid_{}.png'.format(outname, '{}') if not first_set_only else '{}_grid.png'.format(outname)
        grid_image = os.path.join(save_dir, grid_image)
        for set_number in range(num_of_sets):
            start_images = set_number * num_in_set
            total_images = start_images + num_in_set
            fig = plt.figure(1, figsize=(ncols + wider, nrows))
            grid = ImageGrid(
                fig, 111,
                nrows_ncols=(nrows, ncols),  # creates n*m grid of axes
                axes_pad=axes_pad,  # pad between axes in inches
            )
            for i in range(start_images, total_images):
                if (i + 1) > len(data_list):
                    break
                ax = grid[i - start_images]
                ax.imshow(
                    X=data_list[i],
                    cmap=cmap,
                    clim=(min_value, max_value),
                )
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.set_aspect('equal')

            if show:
                plt.show()
            else:
                fig.tight_layout()
                plt.tight_layout()
                plt.savefig(grid_image.format(set_number + 1))

            plt.clf()
            if first_set_only:
                # Save only first set of images:
                break

    # Video - http://stackoverflow.com/a/13983801/4143531:
    ratio = data_list[0].shape[1] / float(data_list[0].shape[0])
    size = 5.0
    fig = plt.figure()
    fig.set_size_inches(ratio * size, size, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    im = ax.imshow(
        data_list[0],
        cmap=cmap,
        clim=(min_value, max_value),
    )

    def update_img(n):
        tmp = data_list[n]
        im.set_data(tmp)
        return im

    ani = animation.FuncAnimation(fig, update_img, len(data_list), interval=1)

    writer = animation.writers['ffmpeg'](fps=10)
    video_file = '{}.mp4'.format(outname)
    ani.save(os.path.join(save_dir, video_file), writer=writer, dpi=100)

    return


def read_single_pic(image_file, x_start, x_len, y_start, y_len, outdir=None, show=False, save=True, cmap='afmhot'):
    """Cut a single image.

    :param image_file:
    :param x_start: start of the cutting area in horizontal dimension.
    :param x_len: length of the cutting area in horizontal dimension.
    :param y_start: start of the cutting area in vertical dimension.
    :param y_len: length of the cutting area in vertical dimension.
    :return:
    """
    # Read the provided image:
    img = uti_io.read_image(image_file, ignore_bottom_limit=True)['raw_image']
    data = np.array(img)

    # Cut necessary area:
    cut_data = data[y_start:y_start + y_len, x_start:x_start + x_len]

    # Rotate the image:
    cut_data = np.rot90(cut_data)

    infile_name, infile_ext = os.path.splitext(os.path.basename(image_file))
    outfile_name = '{}_cut_x={}_y={}_{}x{}{}'.format(infile_name, x_start, y_start, x_len, y_len, infile_ext)

    if show:
        plt.imshow(cut_data, cmap=cmap)
        plt.show()
    else:
        if save:
            # plt.savefig(os.path.join(outdir, outfile_name), pad_inches=0.0, bbox_inches='tight')
            # http://stackoverflow.com/a/978306/4143531 - no white borders:
            plt.imsave(os.path.join(outdir, outfile_name), cut_data, cmap=cmap)

    return cut_data


if __name__ == '__main__':
    base_dir = os.path.abspath('C:\\Users\\Maksim\\Desktop\\2017-02-07 SMI measurements')
    dir_list = [
        'DCM_scan_5.71-6.04',
        'DCM_scan_13.8_14.7',
        'IVU_scan_6.2-6.8_5.77014',
        'IVU_scan_6.2-6.8_14.19',
    ]

    # Cut parameters:
    x_start = 459
    x_len = 75
    y_start = 510
    y_len = 95
    save = False

    # Grid parameters:
    nrows = 10
    ncols = 15
    wider = 3.5
    first_set_only = True

    for d in dir_list:
        image_dir = os.path.join(base_dir, d)

        outdir = os.path.join(os.path.dirname(image_dir), '{}_cut'.format(os.path.basename(image_dir)))
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        # Process in a batch mode:
        cut_images = cut_pictures(image_dir, x_start, x_len, y_start, y_len, search_extention='tif', outdir=outdir,
                                  save=save)
        print('Number of pictures: {}'.format(len(cut_images)))

        plot_grid(cut_images, nrows=nrows, ncols=ncols, save_dir=outdir, wider=wider, first_set_only=first_set_only)

        # Process a single picture:
        # image_file = 'DCMscan_5.71-6.04_6.5mm_451pt_429.tif'
        # read_single_pic(os.path.join(image_dir, image_file), x_start, x_len, y_start, y_len, outdir, show=True)
        # read_single_pic(os.path.join(image_dir, image_file), x_start, x_len, y_start, y_len, outdir, show=False)
