import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, rand_score
from Seeds.it.Brialemi_SRL.dataset.dataset_analisi import DatasetAnalisi 
from io import BytesIO
import base64
import matplotlib.pyplot as plt

class Kmeans:

    def __init__(self, dataset, n_clusters=3, use_pca=True):
        self.val_model = None
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.n_clusters = n_clusters
        self.use_pca = use_pca
        self.set_mod(dataset)

    def set_mod(self, dataset):
        # Target reale: usato solo per valutazione, NON per addestrare KMeans
        y = dataset["classe"]

        # Feature numeriche
        X = dataset.drop(columns=["classe"])
        

        self.feature_columns = X.columns.tolist()

        # Standardizzazione se non usa PCA, altrimenti la PCA include già la standardizzazione
        if not self.use_pca:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            # faimo PCA con 3 componenti principali (o un numero a scelta) e standardizziamo i dati prima di applicare KMeans 
            data_ana = DatasetAnalisi()
            pca_results = data_ana.pca(X, n_components=2, standardize=True)
            X_scaled = pca_results["scores"]
            self.X_pca = np.array(X_scaled)

        # Modello KMeans
        model = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )

        # Addestramento + predizione cluster
        y_pred = model.fit_predict(X_scaled)
        self.y_pred = np.array(y_pred)

        self.model = model
        self.scaler = scaler if not self.use_pca else None

        # Valutazione
        self.val_model = {
            #"predizioni_cluster": y_pred.tolist(),
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
    

    def get_plot(self):

        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(
            self.X_pca[:, 0],
            self.X_pca[:, 1],
            c=self.y_pred,
            cmap="viridis",
            alpha=0.7
        )

        ax.scatter(
            self.model.cluster_centers_[:, 0],
            self.model.cluster_centers_[:, 1],
            marker="X",
            s=300,
            c="red",
            edgecolors="black",
            label="Centroidi"
        )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("KMeans con PCA")

        fig.colorbar(scatter, ax=ax, label="Cluster")

        ax.legend()

        img = BytesIO()

        fig.savefig(
            img,
            format="png",
            bbox_inches="tight"
        )

        img.seek(0)

        encoded = base64.b64encode(
            img.getvalue()
        ).decode()

        plt.close(fig)

        return encoded
