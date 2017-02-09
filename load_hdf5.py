import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_dir = 'C:\\Users\\Maksim\\Desktop'
    hdf5_file = '107fe4cc-5e40-4922-acb4_000000.h5'
    f = h5py.File(os.path.join(data_dir, hdf5_file), 'r')

    data = f['entry']['data']['data']

    shape = data.shape
    num_of_images = shape[0]
    image_size = shape[1:3]

    print('shape:', shape)
    print('num_of_images:', num_of_images)
    print('image_size:', image_size)

    start = 16
    step = 0.5
    end = start + num_of_images * step

    gaps = np.arange(start, end, step)

    outdir = 'out'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # cmap = 'jet'
    cmap = 'gray'
    x_start = 480
    x_end = 717
    y_start = 265
    y_end = 497

    for i in range(num_of_images):
        # plt.imshow(data[i, 273:498, 488:719], cmap='gray')
        # plt.show()
        outfile = 'g_{}.png'.format(gaps[i])
        outpath = os.path.join(outdir, outfile)
        print('File: {}'.format(outpath))
        plt.imsave(outpath, data[i, y_start:y_end, x_start:x_end], cmap=cmap)
        # if i == 3:
        #     break
