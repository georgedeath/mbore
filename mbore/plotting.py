import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_2d_2obj(
    Xtr, Ytr, scalers, ref_point=None, best_inds=None, axdec=None, axobj=None
):
    makeaxes = (axdec is None) or (axobj is None)

    # do some plotting just to visualise what's happening
    if makeaxes:
        fig, (axdec, axobj) = plt.subplots(1, 2, figsize=(9, 4), dpi=120)  # type: ignore

    axdec.scatter(Xtr[:, 0], Xtr[:, 1], c=scalers)
    if best_inds is not None:
        axdec.scatter(
            Xtr[best_inds, 0], Xtr[best_inds, 1], c="r", marker="x", s=10
        )
    axdec.set_xlabel("$x_0$")
    axdec.set_ylabel("$x_1$")

    axobj.scatter(Ytr[:, 0], Ytr[:, 1], c=scalers)
    if best_inds is not None:
        axobj.scatter(
            Ytr[best_inds, 0],
            Ytr[best_inds, 1],
            c="r",
            marker="x",
            s=10,
            label="Best 1/3",
        )
    axobj.set_xlabel("$f_0$")
    axobj.set_ylabel("$f_1$")

    if ref_point is not None:
        axobj.set_xlim([-1, ref_point[0]])
        axobj.set_ylim([-1, ref_point[1]])

    if makeaxes:
        plt.show()


def comparison_boxplot(
    box_data,
    method_names,
    models_and_optimizers,
    model_to_colors,
    title="",
    xlabel="",
    ylabel="",
    savename=None,
    showfig=True,
):
    offset_increment = 1.5
    n_methods = len(method_names)
    xv = np.linspace(0, 1 - 1 / n_methods, n_methods)
    widths = 0.9 * (1 / n_methods)
    labels = list(box_data.keys())

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for offset_idx, key in enumerate(box_data):
        ax.bar(
            x=xv + (offset_increment * offset_idx),
            height=box_data[key],
            width=widths,
            color=list(model_to_colors.values()),
        )
    ax.set_ylim([0, 1])

    # set up the bar labels
    xx = np.arange(0.325, len(labels) * offset_increment, offset_increment)
    ax.set_xticks(xx)
    ax.set_xticklabels(labels)

    # make the fake legend
    lh = []
    for model, opt in models_and_optimizers:
        lh.append(
            Line2D(
                [0],
                [0],
                ls="-",
                c=model_to_colors[(model, opt)],
                label=f"{model} {opt}",
                ms=10,
                alpha=1,
            )
        )

    ax.legend(handles=lh, loc="upper left", ncol=n_methods, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if savename is not None:
        savename = f"{savename:s}.pdf"
        plt.savefig(savename, bbox_inches="tight")
        print(f"Saving: {savename:s}")

    if showfig:
        plt.show()
    else:
        plt.close()

    plt.show()
