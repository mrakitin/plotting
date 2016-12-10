import os

import matplotlib.pyplot as plt
import numpy as np


def plot_image(dat_file, out_file, log_scale, min_value, max_value, show_image):
    list_2d, x, y = prepare_data(dat_file)
    name = '{}.png'.format(out_file)

    # http://stackoverflow.com/questions/16492830/colorplot-of-2d-array-matplotlib:
    fig = plt.figure(figsize=(16, 10))

    ax = fig.add_subplot(111)
    ax.set_title('Intensity distribution (log) - {}'.format(' '.join([x.capitalize() for x in out_file.split('_')])))

    if log_scale:
        plt.imshow(np.log10(list_2d), cmap='gray', clim=(min_value, max_value))
    else:
        plt.imshow(np.log10(list_2d), cmap='gray')

    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')

    if show_image:
        plt.show()
    plt.savefig(name)


def prepare_data(dat_file):
    list_1d, x_range, y_range = _read_data(dat_file)
    list_2d = _convert_1d_to_2d(list_1d, x_range, y_range)

    x = np.linspace(x_range[0], x_range[1], x_range[2])
    y = np.linspace(y_range[0], y_range[1], y_range[2])

    return list_2d, x, y


def _convert_1d_to_2d(list_1d, x_range, y_range):
    tot_len = int(x_range[2] * y_range[2])
    len_1d = len(list_1d)
    if len_1d > tot_len:
        list_1d = np.array(list_1d[0:tot_len])
    elif len_1d < tot_len:
        aux_list = np.zeros(len_1d)
        for i in range(len_1d):
            aux_list[i] = list_1d[i]
        list_1d = np.array(aux_list)
    list_1d = np.array(list_1d)
    list_2d = list_1d.reshape(x_range[2], y_range[2], order='F')
    return list_2d


def _parse_header(row, data_type):
    return data_type(row.split('#')[1].strip())


def _read_data(dat_file, skip_lines=11):
    list_1d = np.loadtxt(dat_file)
    with open(dat_file, 'r') as f:
        content = f.readlines()[:skip_lines]
        x_range = [
            _parse_header(content[4], float),
            _parse_header(content[5], float),
            _parse_header(content[6], int),
        ]
        y_range = [
            _parse_header(content[7], float),
            _parse_header(content[8], float),
            _parse_header(content[9], int),
        ]
    return list_1d, x_range, y_range


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot 2D-intensity distribution')
    parser.add_argument('-d', '--dat_file', dest='dat_file', default='', help='input .dat file')
    parser.add_argument('-o', '--out_file', dest='out_file', default='image',
                        help='output image file name (without extension)')
    parser.add_argument('-l', '--log_scale', dest='log_scale', action='store_true', help='use logarithmic scale')
    parser.add_argument('--min_value', dest='min_value', default=3, help='minimum value for logarithmic scale')
    parser.add_argument('--max_value', dest='max_value', default=15, help='maximum value for logarithmic scale')
    parser.add_argument('-s', '--show_image', dest='show_image', action='store_true', help='show image')

    args = parser.parse_args()

    if not args.dat_file or not os.path.isfile(args.dat_file):
        raise ValueError('No input file found: "{}"'.format(args.dat_file))

    plot_image(dat_file=args.dat_file, out_file=args.out_file, log_scale=args.log_scale, min_value=args.min_value,
               max_value=args.max_value, show_image=args.show_image)
