import pandas as pd
from ucimlrepo import fetch_ucirepo
from it.Brialemi_SRL.dataset.dataset_analisi import DatasetAnalisi
from it.Brialemi_SRL.dataset.grafici import Grafici

class DatasetManager: # è un classe CONTROLLER delle altre classi, non fa niente dal punto di vista di calcoli
    def __init__(self):

        self.__dftrain = self.load_file(
            "C:/Users/alisi/OneDrive/Documenti/GitHub/Kmeans/Seeds/seeds_dataset.txt",
            columns=[
            "area",
            "perimetro",
            "compattezza",
            "lunghezza_kernel",
            "larghezza_kernel",
            "asimmetria",
            "lunghezza_solco"]
        )

        self.__dftest = self.load_file(
            "C:/Users/alisi/OneDrive/Documenti/GitHub/Kmeans/Seeds/seeds_test.txt",
            columns=[
            "classe"]
        )

        self.__data_ana = DatasetAnalisi()
        self.__grafici = Grafici()
 

    def load_file(self, path, columns=None, sep=r"\s+"):
            return pd.read_csv(
                path,
                sep=sep,
                header=None,
                names=columns,
                engine="python"
                )

    def set_data(self, data):
        self.__dataset = data

    def analisi(self): 
        val_nan = self.__data_ana.valori_nulli(self.__dftrain)
        val_strani = self.__data_ana.valori_stringhe(self.__dftrain)
        outliers = self.outlier()
        norm = self.__data_ana.normality(self.__dftrain)
        pca = self.__data_ana.pca(self.__dftrain) 
        return {
            "val_nan": val_nan,
            "val_strani": val_strani,
            "outliers": outliers,  # visto che sono tutti categorici fissi (opzioni) non ha senso parlare di outliers
            "test normalità": norm,
            "pca" : pca
        }

    def outlier(self):
        outl_iqr = self.__data_ana.outliers_iqr_per_col(self.__dftrain)
        outl_zscore = self.__data_ana.outliers_zscore_per_col(self.__dftrain)
        return {
            "outl_iqr": outl_iqr,
            "outl_zscore": outl_zscore
        }

    def grafici(self):              
        correlation = self.__grafici.plot_correlation(self.__dftrain) # impossibile fare su categorici
        pca = self.__data_ana.pca(self.__dftrain)
        grafico_pca = self.__grafici.plot_explained_variance(pca)
        list_hist = []
        for col in self.__dftrain.columns:
            hist = self.__grafici.plot_hist(self.__dftrain, col)
            list_hist.append(hist)
        return {
            "correlation": correlation,
            "hist": list_hist,
            "PCA" : grafico_pca
        }

    def clean(self):
        self.__dftrain = self.__data_ana.clean_data(self.__dftrain)

    def pca_data(self):
        return self.__data_ana.pca(self.__dftrain)

    def stampa(self):
        print(self.__dftrain)

    def correlazione(self):
        return self.__data_ana.correlazione(self.__dftrain)

    def get_datatrain(self):
        return self.__dftrain

    def get_datatest(self):
        return self.__dftest

