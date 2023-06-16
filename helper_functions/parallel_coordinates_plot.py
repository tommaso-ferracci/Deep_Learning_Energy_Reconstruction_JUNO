import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# inspired by https://github.com/gregornickel/pcp, refactored to work with the results dataframes returned by Optuna

def set_ytype(data, ytype, colorbar):
    if ytype == None:
        ytype = [[] for _ in range(len(data.columns))]
    for i in range(len(ytype)):
        if not ytype[i]:
            if type(data.iloc[0, i]) is str:
                ytype[i] = "categorial"
            else:
                ytype[i] = "linear" 
    if colorbar: 
        assert ytype[0] == "linear", "colorbar axis needs to be linear"
    return ytype

def set_ylabels(data, ytype, ylabels):
    if ylabels == None:
        ylabels = [[] for _ in range(len(data.columns))]
    for i in range(len(ylabels)): 
        # generate ylabels for string values
        if not ylabels[i] and ytype[i] == "categorial":
            ylabel = []
            for j in range(len(data)):
                if data.iloc[j, i] not in ylabel:
                    ylabel.append(data.iloc[j, i])
            ylabel.sort()
            if len(ylabel) == 1:
                ylabel.append("")
            ylabels[i] = ylabel
    return ylabels

def replace_str_values(data, ytype, ylabels):
    for i in range(len(ytype)):
        if ytype[i] == "categorial":
            for j in range(len(data)):
                data.iloc[j, i] = ylabels[i].index(data.iloc[j, i])
    return data.to_numpy().T

def set_ylim(data, ylim):
    if ylim == None:
        ylim = [[] for _ in range(data.shape[0])]
    for i in range(len(ylim)):
        if not ylim[i]:
            ylim[i] = [np.min(data[i, :]), np.max(data[i, :])]
            if ylim[i][0] == ylim[i][1]:
                ylim[i] = [ylim[i][0] * 0.95, ylim[i][1] * 1.05]
            if ylim[i] == [0.0, 0.0]:
                ylim[i] = [0.0, 1.0]
    return ylim

def get_performance(data, ylim):
    y_min = ylim[-1][0]
    y_max = ylim[-1][1]
    performance = (np.copy(data[-1, :]) - y_min) / (y_max - y_min)
    return performance

# rescales secondary y-axes to scale of the main y-axis
def rescale_data(data, ytype, ylim):
    min_0 = ylim[0][0]
    max_0 = ylim[0][1]
    scale = max_0 - min_0
    for i in range(1, len(ylim)):
        min_i = ylim[i][0]
        max_i = ylim[i][1]
        if ytype[i] == "log":
            logmin_i = np.log10(min_i)
            logmax_i = np.log10(max_i)
            scale_i = logmax_i - logmin_i
            data[i, :] = ((np.log10(data[i, :]) - logmin_i) / scale_i) * scale + min_0
        else:
            data[i, :] = ((data[i, :] - min_i) / (max_i - min_i)) * scale + min_0
    return data

def get_path(data, i):
    n = data.shape[0] # number of y-axes
    verts = list(zip([x for x in np.linspace(0, n - 1, n * 3 - 2)], 
        np.repeat(data[:, i], 3)[1:-1]))
    codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
    path = Path(verts, codes)
    return path

def plot_parallel_coordinates(data, labels, ytype=None, ylim=None, ylabels=None, figsize=(10,5), 
                              rect=[0.125, 0.1, 0.75, 0.8], curves=True, linewidth=1.0, alpha=1.0, 
                              colorbar=True, colorbar_width=0.02, cmap=plt.get_cmap("inferno_r")):
    """
    Displays a parallel coordinates plot of the hyperparameters search.

    Parameters:
        data (pd.DataFrame): pandas dataframe where the last column is the performance of the trial, the 
                             others represent the various hyperparameters values for the trial
        labels (list): labels for y-axes.
        ytype (list, optional): default "None" allows linear axes for numerical values and categorial 
                                axes for data of type string. If ytype is passed, logarithmic axes are 
                                also possible, example: ["categorial", "linear", "log", [], ...]. Empty 
                                fields must be filled with an empty list []
        ylim (list, optional): custom min and max values for y-axes, example: [[0, 1], [], ...]
        ylabels (list, optional): only use this option if you want to print more categories than you have
                                  in your dataset for categorial axes. Requires ylim to be set correctly.
        figsize: (tuple, optional): width, height in inches
        rect: (list, optional): [left, bottom, width, height], defines the position of the figure on
                                the canvas. 
        curves (bool, optional): if True, B-spline curve is drawn (instead of simple lines)
        linewidth (float, optional): specify width of the curves
        alpha (float, optional): alpha value for transparency of the curves
        colorbar (bool, optional): if True, colorbar is drawn
        colorbar_width (float, optional): defines the width of the colorbar.
        cmap (matplotlib.colors.Colormap, optional): specify color palette for colorbar.
    
    Returns:
        matplotlib.figure.Figure
    """
    [left, bottom, width, height] = rect
    # work on copy as modifications will be made to the dataframe
    data = data.copy(deep=True)

    # setup data
    ytype = set_ytype(data, ytype, colorbar) 
    ylabels = set_ylabels(data, ytype, ylabels)
    data = replace_str_values(data, ytype, ylabels)
    ylim = set_ylim(data, ylim)
    performance = get_performance(data, ylim)
    # notice: rescale_data affects only secondary y-axes
    data = rescale_data(data, ytype, ylim)

    # create figure
    fig = plt.figure(figsize=figsize)
    ax0 = fig.add_axes([left, bottom, width, height])
    axes = [ax0] + [ax0.twinx() for i in range(data.shape[0] - 1)]

    for i in range(data.shape[1]):
        if colorbar:
            color = cmap(performance[i])
        else:
            color = "blue"

        # plot interpolated parabola if requested 
        if curves:
            path = get_path(data, i)
            patch = PathPatch(path, facecolor="None", lw=linewidth, alpha=alpha, 
                    edgecolor=color, clip_on=False)
            ax0.add_patch(patch)
        # otherwise plot simple lines
        else:
            ax0.plot(data[:, i], color=color, alpha=alpha, clip_on=False)

    # format x-axis
    ax0.xaxis.tick_top()
    ax0.xaxis.set_ticks_position("none")
    ax0.set_xlim([0, data.shape[0] - 1])
    ax0.set_xticks(range(data.shape[0]))
    ax0.set_xticklabels(labels, fontsize=12)

    # format y-axis
    for i, ax in enumerate(axes):
        ax.spines["left"].set_position(("axes", 1 / (len(labels) - 1) * i))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.set_ylim(ylim[i])
        if ytype[i] == "log":
            ax.set_yscale("log")
        if ytype[i] == "categorial":
            ax.set_yticks(range(len(ylabels[i])))
        if ylabels[i]:
            ax.set_yticklabels(ylabels[i])
        
    if colorbar:
        bar = fig.add_axes([left + width, bottom, colorbar_width, height])
        norm = matplotlib.colors.Normalize(vmin=ylim[-1][0], vmax=ylim[-1][1])
        matplotlib.colorbar.ColorbarBase(bar, cmap=cmap, norm=norm, 
            orientation="vertical")
        bar.tick_params(size=0)
        bar.set_yticklabels([])

    return fig