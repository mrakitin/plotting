# -*- coding: utf-8 -*-
"""
A utility to plot a grid of images from a set of input data files with intensity from SRW simulations.

Date: 2017-01-27
Author: Maksim Rakitin
"""

from plotting import plot_grid

if __name__ == '__main__':
    dat_dir = 'C:\\Users\\Maksim\\Documents\\Work\\Beamlines\\ESM\\2017-01-27 ESM Diagon simulations\\dat_files'
    plot_grid(dat_dir)
