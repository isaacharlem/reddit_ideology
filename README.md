# Computational Methods for Social Science: Reddit Ideology

**Group Members:** Isaac Harlem, Charlie Eden, Ahmed Ahmed

---

## Three Research Questions

1. **Semantic Diversity Over Time**
   How has the semantic diversity of published conservative and liberal discourse changed over time, as seen through posts on **r/conservative** and **r/liberal**?
2. **Event-Driven Convergence**
   How do major U.S. political events influence the semantic convergence between liberal and conservative discourse, and are these effects symmetric across different events? (Baseline patterns vs. emergent shifts?)
3. **Echo-Chamber Severity**
   Can we use semantic similarity as a proxy to measure the severity of echo chambers over time?

---

## Dataset

We leverage the **Reddit Ideology Database**, which contains a large corpus of posts from ideologically distinct subreddits (including **r/conservative** and **r/liberal**).
Access the data at: [https://data.mendeley.com/datasets/2tdr9sjd83/2](https://data.mendeley.com/datasets/2tdr9sjd83/2)

---

## Instructions

It is highly recommended to use **CUDA** if available, since embedding and clustering steps are compute-intensive.
On CPU/MPS expect a runtime > 60 minutes; on 4× A100 GPUs (80 GB), the full pipeline runs in \~5 minutes.

1. **Clone the repository**

   ```bash
   git clone git@github.com:isaacharlem/reddit_ideology.git
   # or
   git clone https://github.com:isaacharlem/reddit_ideology.git
   ```
2. **Create a Conda environment**

   ```bash
   conda create -n red_id python=3.10
   ```
3. **Activate the environment**

   ```bash
   conda activate red_id
   ```
4. **Install package in editable mode**

   ```bash
   pip install -e .
   ```
5. **Obtain an OpenAI API key**
   Sign up or log in at [OpenAI API](https://openai.com/index/openai-api/) and generate a key with read/write access.
6. **Copy and configure**

   ```bash
   cp config.yaml my_config.yaml
   ```

   Then edit **my\_config.yaml**:

   * Set your `openai.api_key`
   * Adjust `device` (e.g. `cuda`)
   * Modify any other parameters as needed
7. **(Optional) Request SLURM resources**

   ```bash
   srun -p general --gres=gpu:a100:4 --pty --cpus-per-task=32 --mem=200G -t 4:00:00 /bin/bash
   ```
8. **Run the pipeline**

   ```bash
   reddit-ideology --config my_config.yaml run
   ```

All outputs (metrics, plots, logs) will be saved under directories specified in your config.

---

## Pipeline Overview

This CLI performs:

1. **Data Loading & Preprocessing**

   * Loads conservative and liberal subreddit posts
   * Cleans text (tokenization, stopword removal)
2. **Embedding & Topic Modeling**

   * Generates document embeddings
   * Runs UMAP for dimensionality reduction
   * Applies HDBSCAN to find clusters → **topics**
3. **Topic Labeling**

   * Extracts top-N terms per cluster
   * Uses OpenAI to generate human-readable labels
4. **Metrics Computation**

   * **Topic Entropy & Count**: diversity measure over time
   * **Semantic Spread**: dispersion of embeddings (ribbon plots)
   * **Intra-group Similarity**: echo-chamber quantification
   * **Cross-group Similarity**: semantic convergence
5. **Statistical Tests & P-values**

   * **Q1**: Trend in topic entropy vs. year (linear regression)
   * **Q2**: Pre/post event change in cross-group similarity (Welch’s t-test)
   * **Q3**: Intra-group difference (paired t-test)
     **Interpretation**:
   * *p ≤ 0.05* ⇒ statistically significant effect (reject null)
   * *p > 0.05* ⇒ insufficient evidence (fail to reject)
6. **Visualizations**

   * Time series for entropy, topic counts
   * Ribbon plots for semantic spread
   * Cross-group & intra-group similarity plots
   * Top-N topic prevalence trends (separate & combined)

---

## Configuration Example

```yaml
data:
  conservative_path: "data/cons_data.json"
  liberal_path: "data/lib_data.json"
  
embedding:
  model_name: "sentence-transformers/all-mpnet-base-v2"
  batch_size: 256
  device: "cuda"

topic_model:
  method: "cluster"
  cluster:
    umap_neighbors: 20
    umap_min_dist: 0.0
    hdbscan_min_cluster_size: 20

analysis:
  time_interval: "Y"
  top_n: 5

events:
  - name: "Election 2012"
    date: "2012-11-06"
  - name: "Election 2016"
    date: "2016-11-08"
  - name: "Election 2020"
    date: "2020-11-08"
  - name: "George Floyd Murder"
    date: "2020-05-25"

output:
  cache_dir: "results/cache"
  plots_dir: "results/plots"
  metrics_dir: "results/metrics"

openai:
  api_key: "<OPENAI-API-KEY>"
  model: "gpt-4o"                        # or another model
  max_terms: 10                         # how many top terms to send for naming

stats:
  trend_test: "linregress"              # type of test for time‐trend p-values
  prepost_window: 3                     # years before/after for event tests
```

---

## License and Citation

Please cite our project as:

Harlem, I., Eden, C., & Ahmed, A. (2025). *Computational Methods for Social Science: Reddit Ideology*. GitHub repository. [https://github.com/isaacharlem/reddit\_ideology](https://github.com/isaacharlem/reddit_ideology)
