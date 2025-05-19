# File: src/reddit_ideology/cli.py
import click
import pandas as pd
import numpy as np
import scipy.stats as stats
from collections import Counter
from reddit_ideology.openai_utils import init_openai, generate_topic_label
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
    """
    Entry point for the Reddit Ideology CLI.
    """
    ctx.obj = {"config": load_config(config)}

@main.command()
@click.pass_context
def run(ctx):
    """
    Executes the full analysis pipeline:
    - Data loading & preprocessing
    - Embedding & topic modeling with caching
    - Topic labeling via OpenAI
    - Metrics computation (entropy, spread, similarity)
    - Statistical tests (p-values)
    - Visualizations
    """
    cfg = ctx.obj["config"]

    # 1) Parse event dates to pandas.Timestamp
    events = []
    for ev in cfg.get('events', []):
        try:
            dt = pd.to_datetime(ev['date'])
            events.append({'name': ev['name'], 'date': dt})
        except Exception:
            continue

    # 2) Initialize OpenAI if labeling topics
    client = init_openai(cfg.get('openai', {}).get('api_key'))

    # 3) Load and preprocess data
    dl = DataLoader(
        cfg['data']['conservative_path'], cfg['data']['liberal_path']
    )
    cons_df, lib_df = dl.load()
    pp = Preprocessor()
    cons_df = pp.apply(cons_df)
    lib_df = pp.apply(lib_df)

    # 4) Generate embeddings (with cache)
    emb_cfg = cfg['embedding']
    embedder = EmbeddingModel(
        model_name=emb_cfg['model_name'],
        batch_size=emb_cfg['batch_size'],
        device=emb_cfg['device'],
        cache_dir=cfg['output']['cache_dir']
    )
    cons_emb = embedder.embed(cons_df['clean_text'].tolist(), 'conservative')
    lib_emb = embedder.embed(lib_df['clean_text'].tolist(), 'liberal')

    # 5) Topic modeling and labeling
    tm_cfg = cfg['topic_model']['cluster']
    topic_model = EmbeddingClusterTopicModel(
        umap_neighbors=tm_cfg['umap_neighbors'],
        umap_min_dist=tm_cfg['umap_min_dist'],
        hdbscan_min_cluster_size=tm_cfg['hdbscan_min_cluster_size'],
        cache_dir=cfg['output']['cache_dir']
    )
    cons_topics = topic_model.fit(cons_emb, 'conservative')
    lib_topics = topic_model.fit(lib_emb, 'liberal')

    # Extract top terms per topic
    def extract_top_terms(df, topics, top_n):
        term_counts = {}
        for tid in sorted(set(topics)):
            texts = df.loc[topics == tid, 'clean_text']
            words = Counter(' '.join(texts).split())
            term_counts[tid] = [w for w,_ in words.most_common(top_n)]
        return term_counts

    max_terms = cfg.get('openai', {}).get('max_terms', 10)
    # Label topics via OpenAI
    cons_terms = extract_top_terms(cons_df, cons_topics, max_terms)
    cons_labels = {tid: generate_topic_label(client, terms, model=cfg['openai']['model'])
                   for tid, terms in cons_terms.items()}
    lib_terms  = extract_top_terms(lib_df,  lib_topics,  max_terms)
    lib_labels = {tid: generate_topic_label(client, terms, model=cfg['openai']['model'])
                  for tid, terms in lib_terms.items()}

    # 6) Compute metrics
    mc = MetricsCalculator(cfg['output']['metrics_dir'])
    cons_metrics = mc.topic_entropy_and_count(cons_topics, cons_df['timestamp'], freq=cfg['analysis']['time_interval'])
    lib_metrics  = mc.topic_entropy_and_count(lib_topics, lib_df['timestamp'],    freq=cfg['analysis']['time_interval'])
    spread_df    = mc.semantic_spread(
        np.vstack([cons_emb, lib_emb]),
        np.concatenate([cons_topics, lib_topics]),
        pd.concat([cons_df['timestamp'], lib_df['timestamp']]),
        freq=cfg['analysis']['time_interval']
    )
    intra_cons = mc.intra_group_similarity(cons_emb, cons_df['timestamp'], freq=cfg['analysis']['time_interval'])
    intra_lib  = mc.intra_group_similarity(lib_emb,  lib_df['timestamp'],  freq=cfg['analysis']['time_interval'])
    cross_sim  = mc.cross_group_similarity(cons_emb, lib_emb, cons_df['timestamp'], lib_df['timestamp'], freq=cfg['analysis']['time_interval'])

    # 7) Statistical tests & p-values
    # Q1: trend in topic entropy
    years_con = cons_metrics['period'].dt.year.astype(int)
    p_ent_con = stats.linregress(years_con, cons_metrics['entropy']).pvalue
    years_lib = lib_metrics['period'].dt.year.astype(int)
    p_ent_lib = stats.linregress(years_lib, lib_metrics['entropy']).pvalue
    click.echo(f"Q1: Entropy trend p-values => conservative={p_ent_con:.3g}, liberal={p_ent_lib:.3g}")

    # Q2: pre/post event change in cross similarity
    prepost = cfg.get('stats', {}).get('prepost_window', 3)
    for ev in events:
        before = cross_sim[(cross_sim['period'] >= ev['date'] - pd.DateOffset(months=prepost)) & (cross_sim['period'] < ev['date'])]['cross_similarity']
        after  = cross_sim[(cross_sim['period'] > ev['date']) & (cross_sim['period'] <= ev['date'] + pd.DateOffset(months=prepost))]['cross_similarity']
        if len(before) and len(after):
            p = stats.ttest_ind(before, after, equal_var=False).pvalue
            click.echo(f"Q2 ({ev['name']}): cross-sim p-value = {p:.3g}")

    # Q3: paired test for echo chamber difference
    merged = pd.merge(intra_cons.rename(columns={'intra_similarity':'con'}), intra_lib.rename(columns={'intra_similarity':'lib'}), on='period')
    p_echo = stats.ttest_rel(merged['con'], merged['lib']).pvalue
    click.echo(f"Q3: Intra-group similarity difference p-value = {p_echo:.3g}")

    # 8) Visualizations
    viz = Visualizer(cfg['output']['plots_dir'])

    # Diversity plots
    viz.plot_time_series(cons_metrics, 'period', 'entropy', title='Conservative Entropy', filename='cons_entropy.png')
    viz.plot_time_series(lib_metrics,  'period', 'entropy', title='Liberal Entropy',    filename='lib_entropy.png')
    viz.plot_time_series(cons_metrics, 'period', 'topic_count', title='Conservative Topic Count', filename='cons_topic_count.png')
    viz.plot_time_series(lib_metrics,  'period', 'topic_count', title='Liberal Topic Count',    filename='lib_topic_count.png')
    # Spread ribbon
    ribbon = spread_df.groupby('period')['spread'].agg(['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]).reset_index()
    ribbon.columns = ['period','median','q1','q3']
    viz.plot_ribbon(ribbon, 'period','median','q1','q3', title='Semantic Spread', filename='semantic_spread.png')
    # Convergence
    viz.plot_time_series(cross_sim, 'period','cross_similarity', title='Cross-Community Similarity', events=events, filename='cross_similarity.png')
    # Echo chamber
    viz.plot_time_series(intra_cons, 'period','intra_similarity', title='Conservative Echo Chamber', filename='intra_cons.png')
    viz.plot_time_series(intra_lib,  'period','intra_similarity', title='Liberal Echo Chamber',    filename='intra_lib.png')

    # Topic trends
    # build freq tables
    def freq_df(df, topics):
        return (
            pd.DataFrame({'timestamp': df['timestamp'], 'topic': topics})
            .assign(period=lambda d: d['timestamp'].dt.to_period(cfg['analysis']['time_interval']).dt.to_timestamp())
            .groupby(['period','topic']).size().reset_index(name='count')
        )
    cons_freq = freq_df(cons_df, cons_topics)
    lib_freq  = freq_df(lib_df,  lib_topics)

    # Map numeric topic IDs to human-readable labels
    cons_freq['topic_label'] = cons_freq['topic'].map(cons_labels)
    lib_freq['topic_label']  = lib_freq['topic'].map(lib_labels)

    # Get top_n from config, default to 5 if not specified
    top_n = cfg.get('analysis', {}).get('top_n', 5)
    
    # Top N topic trends (per subreddit)
    viz.plot_topic_prevalence(
        cons_freq,
        period_col='period',
        topic_col='topic_label',
        count_col='count',
        top_n=top_n,
        normalize=True,
        title=f'Top {top_n} Conservative Topics Over Time',
        filename=f'cons_top{top_n}_topics.png'
    )
    viz.plot_topic_prevalence(
        lib_freq,
        period_col='period',
        topic_col='topic_label',
        count_col='count',
        top_n=top_n,
        normalize=True,
        title=f'Top {top_n} Liberal Topics Over Time',
        filename=f'lib_top{top_n}_topics.png'
    )

    # Combined trends for conservatives vs liberals
    viz.plot_combined_topic_trends(
        cons_freq,
        lib_freq,
        period_col='period',
        topic_col='topic_label',
        count_col='count',
        top_n=top_n,
        normalize=True,
        title=f'Top {top_n} Topics: Conservative vs Liberal',
        filename=f'combined_top{top_n}_topics.png'
    )

if __name__ == '__main__':
    main()
