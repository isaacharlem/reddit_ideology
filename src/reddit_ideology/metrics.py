import os
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity


class MetricsCalculator:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def topic_entropy_and_count(
        self, topics: np.ndarray, timestamps: pd.Series, freq: str = "Y"
    ) -> pd.DataFrame:
        df = pd.DataFrame({"topic": topics, "timestamp": timestamps})
        df["period"] = df["timestamp"].dt.to_period(freq)
        records = []
        for period, group in df.groupby("period"):
            freqs = group["topic"].value_counts(normalize=True)
            ent = entropy(freqs)
            count = group["topic"].nunique()
            records.append(
                {
                    "period": period.start_time,
                    "entropy": ent,
                    "topic_count": count,
                }
            )
        out = pd.DataFrame(records).sort_values("period")
        out.to_csv(
            os.path.join(self.cache_dir, "topic_entropy_count.csv"), index=False
        )
        return out

    def semantic_spread(
        self,
        embeddings: np.ndarray,
        topics: np.ndarray,
        timestamps: pd.Series,
        freq: str = "Y",
    ) -> pd.DataFrame:
        df = pd.DataFrame({"topic": topics, "timestamp": timestamps})
        df["period"] = df["timestamp"].dt.to_period(freq)
        spreads = []
        for (period, topic), idx in df.groupby(
            ["period", "topic"]
        ).groups.items():
            emb = embeddings[list(idx)]
            centroid = emb.mean(axis=0)
            dist = np.linalg.norm(emb - centroid, axis=1)
            spreads.append(
                {
                    "period": period.start_time,
                    "topic": topic,
                    "spread": dist.mean(),
                }
            )
        spread_df = pd.DataFrame(spreads)
        spread_df.to_csv(
            os.path.join(self.cache_dir, "semantic_spread.csv"), index=False
        )
        return spread_df

    def intra_group_similarity(
        self, embeddings: np.ndarray, timestamps: pd.Series, freq: str = "Y"
    ) -> pd.DataFrame:
        df = pd.DataFrame({"timestamp": timestamps})
        df["period"] = df["timestamp"].dt.to_period(freq)
        records = []
        for period, idx in df.groupby("period").groups.items():
            emb = embeddings[list(idx)]
            centroid = emb.mean(axis=0).reshape(1, -1)
            sims = cosine_similarity(emb, centroid).flatten()
            records.append(
                {"period": period.start_time, "intra_similarity": sims.mean()}
            )
        out = pd.DataFrame(records).sort_values("period")
        out.to_csv(
            os.path.join(self.cache_dir, "intra_similarity.csv"), index=False
        )
        return out

    def cross_group_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
        ts1: pd.Series,
        ts2: pd.Series,
        freq: str = "Y",
    ) -> pd.DataFrame:
        df1 = pd.DataFrame({"timestamp": ts1}).assign(
            period=lambda x: x["timestamp"].dt.to_period(freq)
        )
        df2 = pd.DataFrame({"timestamp": ts2}).assign(
            period=lambda x: x["timestamp"].dt.to_period(freq)
        )
        records = []
        common_periods = set(df1["period"]).intersection(set(df2["period"]))
        for period in sorted(common_periods):
            idx1 = df1[df1["period"] == period].index
            idx2 = df2[df2["period"] == period].index
            sims = cosine_similarity(emb1[idx1], emb2[idx2])
            records.append(
                {"period": period.start_time, "cross_similarity": sims.mean()}
            )
        out = pd.DataFrame(records)
        out.to_csv(
            os.path.join(self.cache_dir, "cross_similarity.csv"), index=False
        )
        return out
