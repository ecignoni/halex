import numpy as np


def set_equalaxes(axes):
    if not isinstance(axes, list) and not isinstance(axes, np.ndarray):
        axes = [axes]

    xlims = np.array([ax.get_xlim() for ax in axes])
    ylims = np.array([ax.get_ylim() for ax in axes])
    xmin = np.min(xlims[:, 0])
    xmax = np.max(xlims[:, 1])
    ymin = np.min(ylims[:, 0])
    ymax = np.max(ylims[:, 1])
    lmin = xmin if xmin < ymin else ymin
    lmax = xmax if xmax > ymax else ymax

    for ax in axes:
        ax.set_xlim(lmin, lmax)
        ax.set_ylim(lmin, lmax)
