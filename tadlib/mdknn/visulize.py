import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def plot_contact_map(mat, ax):
    """
    Plot hi-c contact map.

    Parameters
    ----------
    mat : numpy.ndarray, (ndim=2)
        Interaction matrix of a TAD(region).
    ax : matplotlib.axes.Axes
        The axes to plot the matrix.
    """
    cmap = plt.cm.get_cmap("YlOrRd")
    cmap.set_bad("white")
    cmap.set_under("black")
    im = ax.imshow(mat, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.set_title("")
    plt.colorbar(im, cax=cax)


def plot_compare(compare,
        region="", dot_size=10, dot_color="#0000FF",
        sample1_name="", sample2_name="", **kwargs):
    """
    Plot the MDKNN compare result.

    Parameters
    ----------
    compare : `tadlib.analyze.mdknn.Compare`
        MDKNN comparison object.

    region : str
        The genomic region of compare region, like 'chr1:89455000-89925000'

    dot_size : int
        Size of significant interaction point in contact map.

    dot_color : str
        Color of significant interaction point in contact map.
    """
    kwargs.update("figsize", (20,20))  # set default fig size
    fig, (ax1, ax2) = plt.subplots(1, 2, **kwargs)

    c1 = compare.core1
    c2 = compare.core2

    # plot contact map
    plot_contact_map(np.log2(c1.matrix), ax=ax1)
    ax1.scatter(c1.pos[:, 0], c1.pos[:, 1], s=10, c="#0000FF")
    title1 = sample1_name + "\n" if sample1_name else ""
    title1 += f"AP: {c1.AP}, mean distance: {c1.mean_dist}"
    ax1.set_title(title1)

    plot_contact_map(np.log2(c2.matrix), ax=ax2)
    ax2.scatter(c2.pos[:, 0], c2.pos[:, 1], s=10, c="#0000FF")
    title2 = sample2_name + "\n" if sample2_name else ""
    title2 += f"AP: {c2.AP}, mean distance: {c2.mean_dist}"
    ax2.set_title(title2)

    center_text = region + "\n" if region else ""
    center_text += f"p-value: {compare.pvalue}\ndifference: {compare.difference}"
    fig.text(0.42, 0.27, , fontsize=20)
    return fig

