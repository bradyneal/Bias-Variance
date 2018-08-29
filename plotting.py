"""
Module for plotting experimental results
"""

from __future__ import print_function, division
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.switch_backend('agg')

CAPSIZE = 5
XSCALE_DEF = 'linear'
YSCALE_DEF = 'linear'
GRID_DEF = False


def plot_line(x, y, label=None, xlabel=None,
    ylabel=None, title=None, filename=None, xscale=XSCALE_DEF, yscale=YSCALE_DEF,
    legend_loc='upper right', grid=GRID_DEF):
    '''
    Plot figure with 95% confidence interval, according to Normal distirbution.
    Save figure if filename specified; otherwise, show the figure.
    '''
    plt.plot(x, y, 'o-', label=label)
    run_fig_extras(xlabel, ylabel, title, filename, xscale, yscale, legend_loc, grid)


def plot_line_with_normal_errbars(x, y, y_std, label=None, xlabel=None,
    ylabel=None, title=None, filename=None, xscale=XSCALE_DEF, yscale=YSCALE_DEF,
    legend_loc='upper right', grid=GRID_DEF, elinewidth=None, marker=None):
    '''
    Plot figure with 95% confidence interval, according to Normal distirbution.
    Save figure if filename specified; otherwise, show the figure.
    '''
    if isinstance(y_std, list):
        yerr = [1.96 * std for std in y_std]
    else:
        yerr = 1.96 * y_std
    plt.errorbar(x, y, yerr=yerr, fmt=marker, capsize=CAPSIZE, label=label, elinewidth=elinewidth)
    run_fig_extras(xlabel, ylabel, title, filename, xscale, yscale, legend_loc, grid)


def plot_line_with_errbars(x, y, lowers, uppers, label=None, xlabel=None,
    ylabel=None, title=None, filename=None, xscale=XSCALE_DEF, yscale=YSCALE_DEF,
    legend_loc='upper right', grid=GRID_DEF, elinewidth=None, marker=None):
    '''
    Plot figure with error bars specified by lowers and uppers (more general than above).
    Save figure if filename specified; otherwise, show the figure.
    '''
    plt.errorbar(x, y, yerr=[lowers, uppers], fmt=marker, capsize=CAPSIZE, label=label, elinewidth=elinewidth)
    run_fig_extras(xlabel, ylabel, title, filename, xscale, yscale, legend_loc, grid)


def run_fig_extras(xlabel=None, ylabel=None, title=None, filename=None, xscale=XSCALE_DEF, yscale=YSCALE_DEF, legend_loc='upper right', grid=GRID_DEF):
    '''
    Set details about currently opened figure.
    Save figure if filename specified; otherwise, show the figure.
    '''
    labelsize = 15
    if xlabel:
        plt.xlabel(xlabel, fontsize=labelsize)
    if ylabel:
        plt.ylabel(ylabel, fontsize=labelsize)
    # if title:
    #     plt.title(title)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.grid(grid)
    plt.legend(loc=legend_loc)   # fontsize='x-large', bbox_to_anchor=(1.3, 1.0))
    if filename.endswith('pdf'):
        plt.savefig(filename, format='pdf')
    elif filename:
        plt.savefig(filename)
    else:
        plt.show()
