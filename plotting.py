# -*- coding: utf-8 -*-
"""
2D plotting utility.

Date: 2016-12-10
Author: Maksim Rakitin
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_image(dat_file, out_file, log_scale, manual_scale, min_value, max_value, show_image, cmap):
    list_2d, x_values, y_values = prepare_data(dat_file)
    name = '{}.png'.format(out_file)

    # http://stackoverflow.com/a/3900167/4143531:
    mpl.rcParams.update({'font.size': 18})

    # http://stackoverflow.com/questions/16492830/colorplot-of-2d-array-matplotlib:
    fig = plt.figure(figsize=(16, 10))

    ax = fig.add_subplot(111)
    log_scale_text = ' (logarithmic scale)' if log_scale else ''
    ax.set_title('Intensity Distribution{} - {}'.format(
        log_scale_text,
        ' '.join([x.capitalize() for x in out_file.split('_')]))
    )

    x_units = 'm'
    y_units = 'm'
    x_units_prefix = ''
    y_units_prefix = ''

    unit_prefixes = {
        'P': 1e15,
        'T': 1e12,
        'G': 1e9,
        'M': 1e6,
        'k': 1e3,
        '': 1e0,
        'm': 1e-3,
        '$\mu$': 1e-6,
        'n': 1e-9,
        'p': 1e-12,
        'f': 1e-15,
    }
    x_min = x_values.min()
    x_max = x_values.max()
    for k, v in unit_prefixes.items():
        if v <= abs(x_max - x_min) < v * 1e3:
            x_units_prefix = k
            break
    y_min = y_values.min()
    y_max = y_values.max()
    for k, v in unit_prefixes.items():
        if v <= abs(y_max - y_min) < v * 1e3:
            y_units_prefix = k
            break

    kwargs = {
        'cmap': cmap,
        'clim': None,
        'extent': [
            x_min / unit_prefixes[x_units_prefix], x_max / unit_prefixes[x_units_prefix],
            y_min / unit_prefixes[y_units_prefix], y_max / unit_prefixes[y_units_prefix],
        ],
    }
    if log_scale:
        data = np.log10(list_2d)
        if manual_scale:
            kwargs['clim'] = (float(min_value), float(max_value))
    else:
        data = list_2d

    plt.imshow(data, **kwargs)

    ax.set_aspect('equal')
    # ax.set_aspect('1.25')
    ax.set_xlabel(r'Horizontal Position [{}{}]'.format(x_units_prefix, x_units))
    ax.set_ylabel(r'Vertical Position [{}{}]'.format(y_units_prefix, y_units))

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')

    if show_image:
        plt.show()
    else:
        plt.savefig(name)


def prepare_data(dat_file):
    list_1d, x_range, y_range = _read_data(dat_file)
    list_2d = _convert_1d_to_2d(list_1d, x_range, y_range)

    x = np.linspace(*x_range)
    y = np.linspace(*y_range)

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
    parser.add_argument('-m', dest='manual_scale', action='store_true',
                        help='set the limits of the logarithmic scale manually')
    parser.add_argument('--min_value', dest='min_value', default=3, help='minimum value for logarithmic scale')
    parser.add_argument('--max_value', dest='max_value', default=15, help='maximum value for logarithmic scale')
    parser.add_argument('-s', '--show_image', dest='show_image', action='store_true', help='show image')
    parser.add_argument('-c', '--cmap', dest='cmap', default='gray', help='color map')

    args = parser.parse_args()

    if not args.dat_file or not os.path.isfile(args.dat_file):
        raise ValueError('No input file found: "{}"'.format(args.dat_file))

    plot_image(dat_file=args.dat_file, out_file=args.out_file, log_scale=args.log_scale,
               manual_scale=args.manual_scale, min_value=args.min_value, max_value=args.max_value,
               show_image=args.show_image, cmap=args.cmap)
