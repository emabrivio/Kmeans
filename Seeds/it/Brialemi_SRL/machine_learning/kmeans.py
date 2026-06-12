import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, rand_score


class Kmeans:

    def __init__(self, dataset, n_clusters=3, use_pca=False):
        self.val_model = None
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.n_clusters = n_clusters
        self.use_pca = use_pca
        self.set_mod(dataset)

    def set_mod(self, dataset):
        # Target reale: usato solo per valutazione, NON per addestrare KMeans
        y = dataset["Class"]

        # Feature numeriche
        X = dataset.drop(columns=["Class"])

        self.feature_columns = X.columns.tolist()

        # Standardizzazione se non usa PCA, altrimenti la PCA include già la standardizzazione
        if not self.use_pca:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values

        # Modello KMeans
        model = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )

        # Addestramento + predizione cluster
        y_pred = model.fit_predict(X_scaled)

        self.model = model
        self.scaler = scaler

        # Valutazione
        self.val_model = {
            "predizioni_cluster": y_pred.tolist(),
            "silhouette_score": silhouette_score(X_scaled, y_pred),
            "rand_index": rand_score(y, y_pred),
            "inertia": model.inertia_,
            "centroidi": model.cluster_centers_.tolist()
        }

    def get_val(self):
        return self.val_model

    def prevedi(self, osservazione):

        # se arriva lista di coppie -> dict
        if isinstance(osservazione, list):
            osservazione = dict(osservazione)

        df = pd.DataFrame([osservazione])

        # riallineamento colonne
        df = df.reindex(columns=self.feature_columns, fill_value=0)

        # standardizzazione
        df_scaled = self.scaler.transform(df)

        return self.model.predict(df_scaled)

    def prevedi_csv(self, dataframe):
        df = dataframe.reindex(columns=self.feature_columns, fill_value=0)

        # standardizzazione
        df_scaled = self.scaler.transform(df)

        predizioni = self.model.predict(df_scaled)

        output = pd.DataFrame({
            "Cluster": predizioni
        })

        lista_cluster = predizioni.tolist()

        output.to_csv("csvs/kmeans_prev_test.csv", index=False)

        return lista_cluster