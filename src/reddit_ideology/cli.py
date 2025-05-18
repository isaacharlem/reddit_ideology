import click
import pandas as pd
from reddit_ideology.config import load_config
from reddit_ideology.data_loader import DataLoader
from reddit_ideology.preprocessing import Preprocessor
from reddit_ideology.embedding import EmbeddingModel
from reddit_ideology.topic_model import EmbeddingClusterTopicModel
from reddit_ideology.metrics import MetricsCalculator
from reddit_ideology.visualize import Visualizer


@click.group()
@click.option(
    "--config", "-c", default="config.yaml", help="Path to config file."
)
@click.pass_context
def main(ctx, config):
    ctx.obj = {"config": load_config(config)}


@main.command()
@click.pass_context
def run(ctx):
    cfg = ctx.obj["config"]
    # Load data
    dl = DataLoader(
        cfg["data"]["conservative_path"], cfg["data"]["liberal_path"]
    )
    cons_df, lib_df = dl.load()
    # Preprocess
    pp = Preprocessor()
    cons_df = pp.apply(cons_df)
    lib_df = pp.apply(lib_df)
    # Embed
    emb_cfg = cfg["embedding"]
    embedder = EmbeddingModel(
        model_name=emb_cfg["model_name"],
        batch_size=emb_cfg["batch_size"],
        device=emb_cfg["device"],
        cache_dir=cfg["output"]["cache_dir"],
    )
    cons_emb = embedder.embed(cons_df["clean_text"].tolist(), "conservative")
    lib_emb = embedder.embed(lib_df["clean_text"].tolist(), "liberal")
    # Topic modeling
    tm_cfg = cfg["topic_model"]["cluster"]
    topic_model = EmbeddingClusterTopicModel(
        umap_neighbors=tm_cfg["umap_neighbors"],
        umap_min_dist=tm_cfg["umap_min_dist"],
        hdbscan_min_cluster_size=tm_cfg["hdbscan_min_cluster_size"],
        cache_dir=cfg["output"]["cache_dir"],
    )
    cons_topics = topic_model.fit(cons_emb, "conservative")
    lib_topics = topic_model.fit(lib_emb, "liberal")
    # Metrics
    mc = MetricsCalculator(cfg["output"]["metrics_dir"])
    cons_metrics = mc.topic_entropy_and_count(
        cons_topics, cons_df["timestamp"], freq=cfg["analysis"]["time_interval"]
    )
    lib_metrics = mc.topic_entropy_and_count(
        lib_topics, lib_df["timestamp"], freq=cfg["analysis"]["time_interval"]
    )
    spread_df = mc.semantic_spread(
        np.vstack([cons_emb, lib_emb]),
        np.concatenate([cons_topics, lib_topics]),
        pd.concat([cons_df["timestamp"], lib_df["timestamp"]]),
        freq=cfg["analysis"]["time_interval"],
    )
    intra_cons = mc.intra_group_similarity(
        cons_emb, cons_df["timestamp"], freq=cfg["analysis"]["time_interval"]
    )
    intra_lib = mc.intra_group_similarity(
        lib_emb, lib_df["timestamp"], freq=cfg["analysis"]["time_interval"]
    )
    cross_sim = mc.cross_group_similarity(
        cons_emb,
        lib_emb,
        cons_df["timestamp"],
        lib_df["timestamp"],
        freq=cfg["analysis"]["time_interval"],
    )
    # Visualize
    viz = Visualizer(cfg["output"]["plots_dir"])
    # Question 1: Diversity
    viz.plot_time_series(
        cons_metrics,
        "period",
        "entropy",
        title="Conservative Topic Entropy",
        filename="cons_entropy.png",
    )
    viz.plot_time_series(
        lib_metrics,
        "period",
        "entropy",
        title="Liberal Topic Entropy",
        filename="lib_entropy.png",
    )
    viz.plot_time_series(
        cons_metrics,
        "period",
        "topic_count",
        title="Conservative Topic Count",
        filename="cons_topic_count.png",
    )
    viz.plot_time_series(
        lib_metrics,
        "period",
        "topic_count",
        title="Liberal Topic Count",
        filename="lib_topic_count.png",
    )
    # Ribbon for semantic spread
    ribbon_df = (
        spread_df.groupby("period")["spread"]
        .agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
        .reset_index()
    )
    ribbon_df.columns = ["period", "median", "q1", "q3"]
    viz.plot_ribbon(
        ribbon_df,
        "period",
        "median",
        "q1",
        "q3",
        title="Semantic Spread Over Time",
        filename="semantic_spread.png",
    )
    # Question 2: Convergence with events
    viz.plot_time_series(
        cross_sim,
        "period",
        "cross_similarity",
        title="Cross-community Semantic Similarity",
        events=cfg["events"],
        filename="cross_similarity.png",
    )
    # Question 3: Echo chambers
    viz.plot_time_series(
        intra_cons,
        "period",
        "intra_similarity",
        title="Conservative Intra-group Similarity",
        filename="intra_cons.png",
    )
    viz.plot_time_series(
        intra_lib,
        "period",
        "intra_similarity",
        title="Liberal Intra-group Similarity",
        filename="intra_lib.png",
    )


if __name__ == "__main__":
    main()
