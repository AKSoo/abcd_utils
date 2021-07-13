import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import abcd


def unflatten_tril(flat_tril):
    """
    Convert a flat ndarray to a 2D lower triangle ndarray.

    Params:
        flat_tril: (n*(n+1)/2,) ndarray

    Returns:
        tril: (n, n) ndarray
    """
    dim = int(np.sqrt(flat_tril.shape[0] * 2))
    tril = np.full((dim, dim), np.nan)

    c = 0
    for i in range(dim):
        for j in range(i + 1):
            tril[i, j] = flat_tril[c]
            c += 1

    return tril


def plot_fcon(fcon, labels=None, ax=None, **kwargs):
    """
    Plot a functional connectivity matrix.

    Params:
        fcon: (n*(n+1)/2,) ndarray
        labels: list of n network labels. Default uses abcd.FCON keys
            without None and 'RST' for retrosplenial temporal
        ax: Axes to plot on. If None, creates a new one.
        **kwargs: Arguments for seaborn.heatmap

    Returns:
        display: Axes from seaborn.heatmap
    """
    if ax is None:
        fig, ax = plt.subplots()

    if labels is None:
        labels = []
        for code in abcd.FCON.keys():
            if code == 'n':
                continue
            elif code == 'rspltp':
                labels.append('RST')
            else:
                labels.append(code.upper())

    fcon2d = pd.DataFrame(unflatten_tril(fcon), index=labels, columns=labels)

    return sns.heatmap(fcon2d, ax=ax, square=True, **kwargs)
