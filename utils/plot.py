import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 11
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"


def plot_confusion_matrix(y_true, y_pred, output_plot):
    fig, ax = plt.subplots(figsize=(8, 6))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        ax=ax,
        cmap="Blues",
        colorbar=False
    )
    ax.grid(False)
    ax.set_title("")

    ax.set_xlabel("Previsto", fontsize=11, fontname="Arial", color="black")
    ax.set_ylabel("Verdadeiro", fontsize=11, fontname="Arial", color="black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    ax.tick_params(axis="both", labelsize=11, width=1.5, colors="black")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Arial")

    for text in disp.text_.ravel():
        text.set_fontname("Arial")
        text.set_fontsize(11)
        text.set_color("black")

    fig.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_precision_recall(y_true, y_score, output_plot):
    fig, ax = plt.subplots(figsize=(8, 6))

    PrecisionRecallDisplay.from_predictions(
        y_true,
        y_score,
        ax=ax
    )

    ax.grid(False)
    ax.set_title("")
    ax.set_xlabel("Revocação", fontsize=11, fontname="Arial", color="black")
    ax.set_ylabel("Precisão", fontsize=11, fontname="Arial", color="black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    ax.tick_params(axis="both", labelsize=11, width=1.5, colors="black")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Arial")

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.15),
        frameon=False,
        ncol=1,
        labelspacing=0.3,
        handletextpad=0.5,
        borderaxespad=0.0,
        prop={"family": "Arial", "size": 11}
    )

    fig.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true, y_score, output_plot):
    fig, ax = plt.subplots(figsize=(8, 6))

    RocCurveDisplay.from_predictions(
        y_true,
        y_score,
        ax=ax
    )

    ax.grid(False)
    ax.set_title("")
    ax.set_xlabel("Taxa de Falsos Positivos", fontsize=11, fontname="Arial", color="black")
    ax.set_ylabel("Taxa de Verdadeiros Positivos", fontsize=11, fontname="Arial", color="black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    ax.tick_params(axis="both", labelsize=11, width=1.5, colors="black")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Arial")

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.15),
        frameon=False,
        ncol=1,
        labelspacing=0.3,
        handletextpad=0.5,
        borderaxespad=0.0,
        prop={"family": "Arial", "size": 11}
    )
    fig.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)