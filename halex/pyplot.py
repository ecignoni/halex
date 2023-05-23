from __future__ import annotations
from typing import Tuple
from matplotlib.figure import Figure

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _sanify_axes(axes):
    if not isinstance(axes, list) and not isinstance(axes, np.ndarray):
        axes = [axes]
    return axes


def set_equalaxes(axes):
    axes = _sanify_axes(axes)
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


def set_equalticks(axes, ticks):
    axes = _sanify_axes(axes)
    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)


def plot_distribution_mo_energies(
    mo_energy_pred: torch.Tensor,
    mo_energy_true: torch.Tensor,
    mo_occ: torch.Tensor,
    n: int = 3,
    kind: str = "occvir",
) -> Tuple[Figure, plt.Axes]:
    indices = torch.arange(len(mo_occ))

    if kind == "occvir":
        indices = torch.concatenate(
            (indices[mo_occ == 2][-(n + 1) :], indices[mo_occ == 0][:n])
        )
    elif kind == "occ":
        indices = indices[mo_occ == 2][-n:]
    elif kind == "vir":
        indices = indices[mo_occ == 0][:n]

    mo_energy_pred = mo_energy_pred.clone()[:, indices]
    mo_energy_true = mo_energy_true.clone()[:, indices]
    data = torch.row_stack((mo_energy_true, mo_energy_pred)).detach().numpy()
    columns = [f"MO {i:d}" for i in indices]
    df = pd.DataFrame(data=data, columns=columns)
    df["kind"] = np.concatenate(
        (
            np.full(len(mo_energy_true), "Target"),
            np.full(len(mo_energy_pred), "Prediction"),
        )
    )
    df = pd.melt(df, id_vars="kind", var_name="MO", value_name="Energy")

    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), dpi=120)

    sns.violinplot(data=df, x="MO", y="Energy", hue="kind", split=True, ax=ax)

    fig.tight_layout()
