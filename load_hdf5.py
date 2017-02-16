import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_hdf5(data_dir, hdf5_file):
    f = h5py.File(os.path.join(data_dir, hdf5_file), 'r')

    data = f['entry']['data']['data']

    shape = data.shape
    num_of_images = shape[0]
    image_size = shape[1:3]

    return {
        'data': data,
        'num_of_images': num_of_images,
        'image_size': image_size,
    }


def get_gaps_range(num_of_images, start=16, step=0.5):
    end = start + num_of_images * step
    gaps = np.arange(start, end, step)
    return gaps

def gap_to_num(gap, gaps_list):
    for i in range(len(gaps_list)):
        if abs(gaps_list[i] - gap) < 1e-8:
            return i

def cut_data(data, image_number, x_start=483, x_end=720, y_start=265, y_end=499):
    return data[image_number, y_start:y_end, x_start:x_end]


if __name__ == '__main__':
    data_dir = 'C:\\Users\\Maksim\\Desktop'
    hdf5_file = '107fe4cc-5e40-4922-acb4_000000.h5'
    d = load_hdf5(data_dir=data_dir, hdf5_file=hdf5_file)
    data = d['data']
    num_of_images = d['num_of_images']

    outdir = 'hdf5_to_images'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    gaps = get_gaps_range(num_of_images, start=16, step=0.5)

    for i in range(num_of_images):
        outfile = 'g_{}.png'.format(gaps[i])
        outpath = os.path.join(outdir, outfile)
        print('File: {}'.format(outpath))
        plt.imsave(outpath, cut_data(data=data, image_number=i), cmap='gray')
