import matplotlib.pyplot as plt
from matplotlib import rcParams

import config as CFG


def create_3_panel_plot(
    sup_title,
    title_1,
    title_2,
    title_3,
    x_lim=[-4, 4],
    y_lim=[-4, 4],
    z_lim=[-4, 4],
    show_ticks=False,
    show_labels=False,
    tick_label_size=None,
    label_font_size=6,
):
    """
    Helper Function to draw 3 Plots in a single image
    """
    fig = plt.figure(figsize=(12, 3), dpi=200)

    fig.suptitle(sup_title)
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    ax.set_title(title_1, fontsize=6)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)

    rcParams["xtick.color"] = CFG.axis_tickcolor
    rcParams["ytick.color"] = CFG.axis_tickcolor
    rcParams["axes.labelcolor"] = CFG.axis_tickcolor
    rcParams["axes.edgecolor"] = CFG.axis_tickcolor

    if not show_ticks:
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.zaxis.set_ticklabels([])

    if tick_label_size is not None:
        ax.xaxis.set_tick_params(labelsize=tick_label_size)
        ax.yaxis.set_tick_params(labelsize=tick_label_size)
        ax.zaxis.set_tick_params(labelsize=tick_label_size)

    if show_labels:
        ax.set_xlabel("$x$ – Ground CS", labelpad=-15, fontsize=label_font_size)
        ax.set_ylabel("$y$ – Ground CS", labelpad=-15, fontsize=label_font_size)
        ax.set_zlabel("$z$ – Ground CS", labelpad=-15, fontsize=label_font_size)

    ax.xaxis._axinfo["grid"].update({"color": CFG.grid_color})
    ax.yaxis._axinfo["grid"].update({"color": CFG.grid_color})
    ax.zaxis._axinfo["grid"].update({"color": CFG.grid_color})
    ax.zaxis._axinfo["tick"].update({"color": CFG.grid_color})
    ax.zaxis._axinfo["label"].update({"color": CFG.axis_tickcolor})

    ax.xaxis.pane.set_edgecolor(CFG.pane_edge_color)
    ax.yaxis.pane.set_edgecolor(CFG.pane_edge_color)
    ax.zaxis.pane.set_edgecolor(CFG.pane_edge_color)
    # ax.xaxis.pane.set_linewidth(2)
    # ax.yaxis.pane.set_linewidth(2)
    # ax.zaxis.pane.set_linewidth(2)

    ax1 = fig.add_subplot(1, 4, 2, projection="3d")
    ax1.set_title(title_2, fontsize=6)

    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)
    ax1.set_zlim(z_lim)

    if not show_ticks:
        ax1.axes.xaxis.set_ticklabels([])
        ax1.axes.yaxis.set_ticklabels([])
        ax1.axes.zaxis.set_ticklabels([])

    if tick_label_size is not None:
        ax1.xaxis.set_tick_params(labelsize=tick_label_size)
        ax1.yaxis.set_tick_params(labelsize=tick_label_size)
        ax1.zaxis.set_tick_params(labelsize=tick_label_size)

    if show_labels:
        ax1.set_xlabel("$x$ – Ground CS", labelpad=-15, fontsize=label_font_size)
        ax1.set_ylabel("$y$ – Ground CS", labelpad=-15, fontsize=label_font_size)
        ax1.set_zlabel("$z$ – Ground CS", labelpad=-15, fontsize=label_font_size)

    ax1.xaxis._axinfo["grid"].update({"color": CFG.grid_color})
    ax1.yaxis._axinfo["grid"].update({"color": CFG.grid_color})
    ax1.zaxis._axinfo["grid"].update({"color": CFG.grid_color})
    ax1.zaxis._axinfo["tick"].update({"color": CFG.grid_color})
    ax1.zaxis._axinfo["label"].update({"color": CFG.axis_tickcolor})

    ax1.xaxis.pane.set_edgecolor(CFG.pane_edge_color)
    ax1.yaxis.pane.set_edgecolor(CFG.pane_edge_color)
    ax1.zaxis.pane.set_edgecolor(CFG.pane_edge_color)
    # ax1.xaxis.pane.set_linewidth(2)
    # ax1.yaxis.pane.set_linewidth(2)
    # ax1.zaxis.pane.set_linewidth(2)

    ax2 = fig.add_subplot(1, 4, 3, projection="3d")
    ax2.set_title(title_3, fontsize=6)

    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)
    ax2.set_zlim(z_lim)

    if not show_ticks:
        ax2.axes.xaxis.set_ticklabels([])
        ax2.axes.yaxis.set_ticklabels([])
        ax2.axes.zaxis.set_ticklabels([])

    if tick_label_size is not None:
        ax2.xaxis.set_tick_params(labelsize=tick_label_size)
        ax2.yaxis.set_tick_params(labelsize=tick_label_size)
        ax2.zaxis.set_tick_params(labelsize=tick_label_size)

    if show_labels:
        ax2.set_xlabel("$x$ – Ground CS", labelpad=-15, fontsize=label_font_size)
        ax2.set_ylabel("$y$ – Ground CS", labelpad=-15, fontsize=label_font_size)
        ax2.set_zlabel("$z$ – Ground CS", labelpad=-15, fontsize=label_font_size)

    ax2.xaxis._axinfo["grid"].update({"color": CFG.grid_color})
    ax2.yaxis._axinfo["grid"].update({"color": CFG.grid_color})
    ax2.zaxis._axinfo["grid"].update({"color": CFG.grid_color})
    ax2.zaxis._axinfo["tick"].update({"color": CFG.grid_color})
    ax2.zaxis._axinfo["label"].update({"color": CFG.axis_tickcolor})
    ax2.xaxis.pane.set_edgecolor(CFG.pane_edge_color)
    ax2.yaxis.pane.set_edgecolor(CFG.pane_edge_color)
    ax2.zaxis.pane.set_edgecolor(CFG.pane_edge_color)
    # ax2.xaxis.pane.set_linewidth(2)
    # ax2.yaxis.pane.set_linewidth(2)
    # ax2.zaxis.pane.set_linewidth(2)

    return (ax, ax1, ax2)
