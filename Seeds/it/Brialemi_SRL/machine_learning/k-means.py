import numpy as np
import pandas as pd 

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.preprocessing import StandardScaler


class KMeansModel:

    def __init__(self, n_clusters=3, random_state=42, scale=True):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scale = scale

        self.model = None
        self.scaler = StandardScaler() if scale else None

    # -------------------------
    # PREPROCESSING
    # -------------------------
    def _prepare_data(self, X):
        X = np.array(X)

        if self.scale:
            X = self.scaler.fit_transform(X)

        return X

    # -------------------------
    # TRAIN
    # -------------------------
    def fit(self, X):
        X = self._prepare_data(X)

        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )

        self.model.fit(X)
        return self

    # -------------------------
    # PREDICTION
    # -------------------------
    def predict(self, X):
        X = self._prepare_data(X)
        return self.model.predict(X)

    # -------------------------
    # METRICHE
    # -------------------------
    def evaluate(self, X):
        X = self._prepare_data(X)

        labels = self.model.labels_

        metrics = {
            "inertia": float(self.model.inertia_),
            "silhouette_score": float(silhouette_score(X, labels)),
            "davies_bouldin_score": float(davies_bouldin_score(X, labels)),
            "calinski_harabasz_score": float(calinski_harabasz_score(X, labels)),
            "n_clusters": int(self.n_clusters)
        }

        return metrics

    # -------------------------
    # INFO CLUSTER
    # -------------------------
    def cluster_summary(self, X):
        X = self._prepare_data(X)
        labels = self.model.labels_

        df = pd.DataFrame(X)
        df["cluster"] = labels

        summary = df.groupby("cluster").mean().to_dict()

        return summary