import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from tayloranalysis.utils import save_item, save_stacked_single_value_tc_point

matplotlib.rc("font", size=16, family="serif")
lw, markeredgewidth = 3, 3


def checkpoints_from_single_values(tc_object, variable_names=None, path="./tc_training.pdf"):
    """
    Plot saved checkpoints. Only works for summarization functions where a single value (i.e. mean) is returned.

    Args:
        variable_names (list[str]): Contains the (LaTeX) type names for the plots. If not
                                        otherwise specified defaults are used ["x_1", "x_2", ...].
        path (str) or (list[str]): /path/to/save/plot.pdf or ["/path/to/save/plot.pdf", "/path/to/save/plot.png"]
    """
    # TODO: Redesign it and access only publically available attributes

    variable_names = variable_names if variable_names else tc_object.variable_names

    for node, dataframe in tc_object._tc_points.items():
        dataframe = dataframe.astype(float)
        fig_and_ax = [plt.subplots(1, 1, figsize=(10, 7)) for _ in range(tc_object.derivation_order + 1)]
        fig, ax = tuple(zip(*fig_and_ax))  # 0: all, 1: first order, 2: second order...

        for column in dataframe.columns:
            _label = ",".join(np.array(variable_names)[np.array(column)])
            _label = f"$<t_{{{_label}}}>$"
            ax[0].plot(dataframe[column], label=_label, lw=lw)
            ax[len(column)].plot(dataframe[column], label=_label, lw=lw)

        for _ax in ax:
            _ax.legend(loc="upper left", bbox_to_anchor=(1.04, 1))
            _ax.set_xlabel("Epoch", loc="right", fontsize=13)
            _ax.set_ylabel("$<t_i>$", loc="top", fontsize=13)
            _ax.yaxis.set_tick_params(which="both", right=True, direction="in")
            _ax.xaxis.set_tick_params(which="both", top=True, direction="in")

        prefix = [""] + [f"order_{i+1}" for i in range(tc_object.derivation_order)]
        prefix_node = "_".join(map(str, node)) if isinstance(node, tuple) else node
        prefix = [f"node_{prefix_node}_{pre}" for pre in prefix]
        for _fig, _pref in zip(fig, prefix):
            save_item(_fig, path, prefix=_pref)

        plt.close("all")


def taylor_coefficients_from_single_values(
    tc_object,
    variable_names=None,
    sorted=True,
    number_of_tc_per_plot=20,
    path="./coefficients.pdf",
):
    """
    Plot taylorcoefficients for current weights of the model. . Only works for summarization functions where a
    single value (i.e. mean) is returned.

    Args:
        variable_names (list[str]): Contains the (LaTeX) type names for the plots. If not
                                    otherwise specified defaults are used ["x_1", "x_2", ...].
        sorted (bool): Sort the computed Taylor coefficients based on their numerical value.
        number_of_tc_per_plot (int): number of drawn taylor coefficients inside one plot. If the number of
                                     taylor coefficients is greater than number_of_tc_per_plot multiple
                                     plots are created.
        path (str) or (list[str]): /path/to/save/plot.pdf or ["/path/to/save/plot.pdf", "/path/to/save/plot.png"]
    """
    # TODO: Redesign it and access only publically available attributes

    variable_names = variable_names or [f"x_{idx}" for idx in tc_object._variable_idx]

    save_stacked_single_value_tc_point(tc_object=tc_object, variable_names=variable_names, path=path)

    for node, _dataframe in tc_object._tc_point.items():
        _dataframe = _dataframe.astype(float)
        if sorted:
            _dataframe.sort_values(by=0, axis=1, ascending=0, inplace=True, key=abs)

        m, n = _dataframe.shape[1], number_of_tc_per_plot
        splits = [np.arange(m)[i : i + n] for i in range(0, m, n)]

        prefix = f'node_{"_".join(map(str, node)) if isinstance(node, tuple) else node}'
        directory, filename = tuple(os.path.split(path if isinstance(path, str) else path[0]))
        filename = f"{prefix}_{os.path.splitext(filename)[0]}_combined.pdf"
        combined_pdf = os.path.join(directory, filename)

        with PdfPages(combined_pdf) as pdf:
            figs = []
            leftpads, rightpads = [], []
            for split_idx, split in enumerate(splits):
                fig, ax = plt.subplots(1, 1, figsize=(10, 7))
                ylabels = []
                for idx, column in enumerate(_dataframe.columns[split][::-1]):
                    _label = ",".join(np.array(variable_names)[np.array(column)])
                    ylabels.append(f"$<t_{{{_label}}}>$")
                    ax.plot(_dataframe.loc[0][column], idx, marker="+", color="black", markersize=10, markeredgewidth=markeredgewidth)

                ax.set_xlabel("$<t_i>$")
                ax.set_ylim(ax.get_ylim())
                if not tc_object._apply_abs:
                    xtick = abs(np.array(list(ax.get_xlim()))).max()
                    xmargin = 2 * xtick * plt.margins()[1]
                    ax.set_xlim(-xtick - xmargin, xtick + xmargin)
                    ax.set_xticks([-xtick, 0, +xtick])
                    ax.vlines(0, *ax.get_ylim(), alpha=0.125, color="grey", ls="-", lw=1)
                ax.set_yticks(list(range(idx + 1)))
                ax.set_yticklabels(ylabels, ha="right", rotation_mode="anchor")
                ax.grid(axis="y", alpha=0.25)

                plt.tight_layout()
                prefix = f'node_{"_".join(map(str, node)) if isinstance(node, tuple) else node}'
                postfix = f"{split_idx}" if len(splits) > 1 else None
                save_item(fig, path, prefix=prefix, postfix=postfix)
                figs.append(fig)
                leftpads.append(ax._originalPosition.get_points()[0][0])
                rightpads.append(ax._originalPosition.get_points()[1][0])
            for fig in figs:
                fig.subplots_adjust(left=max(leftpads), right=min(rightpads))
                pdf.savefig(fig)
            plt.close("all")
