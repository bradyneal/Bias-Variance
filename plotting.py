"""
Module for plotting experimental results
"""

from __future__ import print_function, division
import matplotlib.pyplot as plt
plt.switch_backend('agg')

CAPSIZE = 5
XSCALE_DEF = 'linear'
YSCALE_DEF = 'linear'
GRID_DEF = False


def plot_line_with_normal_errbars(x, y, y_std, xlabel=None, ylabel=None, title=None, filename=None, xscale=XSCALE_DEF, yscale=YSCALE_DEF, grid=GRID_DEF):
    '''
    Plot figure with 95% confidence interval, according to Normal distirbution.
    Save figure if filename specified; otherwise, show the figure.
    '''
    if isinstance(y_std, list):
        yerr = [2 * std for std in y_std]
    else:
        yerr = 2 * y_std
    plt.figure()
    plt.errorbar(x, y, yerr=yerr, capsize=CAPSIZE)
    run_fig_extras(xlabel, ylabel, title, filename, xscale, yscale, grid)


def run_fig_extras(xlabel=None, ylabel=None, title=None, filename=None, xscale=XSCALE_DEF, yscale=YSCALE_DEF, grid=GRID_DEF):
    '''Set details about currently opened figure and show or save the figure'''
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.grid(grid)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()
