import os
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    def __init__(self, plots_dir: str):
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_time_series(
        self,
        df,
        x: str,
        y: str,
        hue: str = None,
        title: str = "",
        events: list = None,
        filename: str = "",
    ):
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x=x, y=y, hue=hue, marker="o")
        if events:
            for ev in events:
                plt.axvline(ev["date"], linestyle="--", color="gray")
                plt.text(
                    ev["date"],
                    plt.ylim()[1],
                    ev["name"],
                    rotation=90,
                    verticalalignment="bottom",
                )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, filename))
        plt.close()

    def plot_ribbon(
        self,
        df,
        x: str,
        y: str,
        lower: str,
        upper: str,
        title: str,
        filename: str,
    ):
        plt.figure(figsize=(12, 6))
        plt.plot(df[x], df[y], marker="o")
        plt.fill_between(df[x], df[lower], df[upper], alpha=0.3)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, filename))
        plt.close()

    def plot_heatmap(
        self, matrix, x_labels, y_labels, title: str, filename: str
    ):
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matrix, xticklabels=x_labels, yticklabels=y_labels, cmap="viridis"
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, filename))
        plt.close()
