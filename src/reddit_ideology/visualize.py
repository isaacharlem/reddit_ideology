import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
                # Convert event date to datetime if needed
                date = ev.get('date')
                try:
                    date_dt = pd.to_datetime(date)
                except Exception:
                    continue
                plt.axvline(x=date_dt, linestyle="--", color="gray")
                plt.text(
                    date_dt,
                    plt.ylim()[0] - 0.01,
                    ev.get('name', ''),
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

    def plot_topic_prevalence(
        self,
        df: pd.DataFrame,
        period_col: str = 'period',
        topic_col: str = 'topic',
        count_col: str = 'count',
        group_col: str = None,
        top_n: int = 10,
        normalize: bool = True,
        title: str = '',
        filename: str = '',
    ):
        """
        Plot top_n topics’ prevalence over time. If normalize=True, uses proportions.
        df must contain columns: period, topic, count, and optionally group_col.
        """
        data = df.copy()
        # Normalize counts to proportions per period (and group if provided)
        if normalize:
            if group_col:
                data['prop'] = data.groupby([group_col, period_col])[count_col] \
                    .transform(lambda x: x / x.sum())
            else:
                data['prop'] = data.groupby(period_col)[count_col] \
                    .transform(lambda x: x / x.sum())
            value_col = 'prop'
        else:
            value_col = count_col

        # Determine top topics overall (or per group if provided)
        if group_col:
            # sum proportions or counts across both groups for ranking
            ranking = data.groupby(topic_col)[value_col].sum().nlargest(top_n).index
        else:
            ranking = data.groupby(topic_col)[value_col].sum().nlargest(top_n).index
        plot_df = data[data[topic_col].isin(ranking)]

        plt.figure(figsize=(12, 6))
        if group_col:
            sns.lineplot(
                data=plot_df,
                x=period_col,
                y=value_col,
                hue=topic_col,
                style=group_col,
                markers=True,
                dashes=False,
                errorbar=None
            )
        else:
            sns.lineplot(
                data=plot_df,
                x=period_col,
                y=value_col,
                hue=topic_col,
                marker='o',
                errorbar=None
            )
        plt.title(title or f'Top {top_n} Topics Over Time')
        plt.tight_layout()
        if filename:
            plt.savefig(os.path.join(self.plots_dir, filename))
        plt.close()

    def plot_combined_topic_trends(
        self,
        df_cons: pd.DataFrame,
        df_lib: pd.DataFrame,
        period_col: str = 'period',
        topic_col: str = 'topic',
        count_col: str = 'count',
        top_n: int = 10,
        normalize: bool = True,
        title: str = '',
        filename: str = '',
    ):
        """
        Plot combined topic trends for conservatives vs liberals on the same chart.
        Expects separate DataFrames for each group with identical schema.
        """
        df_cons = df_cons.copy()
        df_lib = df_lib.copy()
        df_cons['group'] = 'conservative'
        df_lib['group'] = 'liberal'
        combined = pd.concat([df_cons, df_lib], ignore_index=True)
        self.plot_topic_prevalence(
            combined,
            period_col=period_col,
            topic_col=topic_col,
            count_col=count_col,
            group_col='group',
            top_n=top_n,
            normalize=normalize,
            title=title or f'Top {top_n} Topics: Conservative vs Liberal',
            filename=filename
        )

