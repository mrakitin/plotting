# plotting
Plotting scripts and utilities

Options:
```
$ python plotting.py -h
usage: plotting.py [-h] [-d DAT_FILE] [-o OUT_FILE] [-l] [-m]
                   [--min_value MIN_VALUE] [--max_value MAX_VALUE] [-s]
                   [-c CMAP]

Plot 2D-intensity distribution

optional arguments:
  -h, --help            show this help message and exit
  -d DAT_FILE, --dat_file DAT_FILE
                        input .dat file
  -o OUT_FILE, --out_file OUT_FILE
                        output image file name (without extension)
  -l, --log_scale       use logarithmic scale
  -m                    set the limits of the logarithmic scale manually
  --min_value MIN_VALUE
                        minimum value for logarithmic scale
  --max_value MAX_VALUE
                        maximum value for logarithmic scale
  -s, --show_image      show image
  -c CMAP, --cmap CMAP  color map
```

Plot the intensity distribution in the logarithmic scale and show the resulted image:
```bash
$ python plotting.py -d res_int_pr_se.dat -l -m --min_value=3.0 --max_value=15 -s
```

Plot the intensity distribution in the logarithmic scale and save the resulted file to the current dir:
```bash
$ python plotting.py -d res_int_pr_se.dat -l -m --min_value=3.0 --max_value=15
```
