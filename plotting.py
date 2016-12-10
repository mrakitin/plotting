import matplotlib.pyplot as plt
import numpy as np


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
    samples = {
        # 'thin_sample': 'res_int_pr_se-thin_sample.dat',
        # 'thin_sample_ideal': 'res_int_pr_se-thin_sample-ideal.dat',
        # 'thick_sample': 'res_int_pr_se-thick_sample.dat',
        # 'thin_sample_rings': 'res_int_pr_se-thin_sample-rings.dat',
        'thin_sample_ideal_rings': 'res_int_pr_se-thin_sample-ideal_rings.dat',
        # 'thick_sample_rings': 'res_int_pr_se-thick_sample-rings.dat',
    }

    for s in samples.keys():
        sample = samples[s]
        list_2d, x, y = prepare_data(sample)
        name = '{}.png'.format(s)

        # http://stackoverflow.com/questions/16492830/colorplot-of-2d-array-matplotlib:
        fig = plt.figure(figsize=(16, 10))

        ax = fig.add_subplot(111)
        ax.set_title('Intensity distribution (log) - {}'.format(' '.join([x.capitalize() for x in s.split('_')])))
        plt.imshow(np.log10(list_2d), cmap='gray', clim=(3.0, 16.0))
        # plt.imshow(np.log10(list_2d), cmap='gray')
        ax.set_aspect('equal')

        cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.patch.set_alpha(0)
        cax.set_frame_on(False)
        plt.colorbar(orientation='vertical')

        # plt.show()
        plt.savefig(name)
