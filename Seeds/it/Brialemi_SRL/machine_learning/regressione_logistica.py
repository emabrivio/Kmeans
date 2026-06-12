from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

class RegLogistica:

    def __init__(self ,data):
        self.val_model = None
        self.set_mod(data)
    
    def set_mod(self, data):
        # Variabili esplicative e target
        y = data["Survived"]
        
        # feature categoriche
        X = pd.get_dummies(
        data.drop(columns=["Survived"]),
        drop_first=True
        )

        self.feature_columns = X.columns.tolist()

        # Suddivisione train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        # Modello
        model = LogisticRegression(max_iter=1000)
        # Addestramento
        model.fit(X_train, y_train)

        self.model= model 
        y_pred = model.predict(X_test)

        coeff_df = pd.DataFrame({
            "variabile": X.columns,
            "coefficiente": self.model.coef_[0]
            })

        # ordinamento per importanza assoluta
        coeff_df["abs_coeff"] = coeff_df["coefficiente"].abs()
        coeff_df = coeff_df.sort_values("abs_coeff", ascending=False)

        self.val_model = {
            "predizioni": y_pred.tolist(),
            "coeff": coeff_df[["variabile", "coefficiente"]].to_dict("records"),
            "intercetta": float(self.model.intercept_[0]),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        }

    def get_val(self):
            return self.val_model
    
    def prevedi(self, osservazione):
    
        # se arriva lista di coppie -> dict
        if isinstance(osservazione, list):
            osservazione = dict(osservazione)

        df = pd.DataFrame([osservazione])

        # one-hot encoding identico al training
        df = pd.get_dummies(df)

        # riallineamento colonne (PASSAGGIO FONDAMENTALE)
        df = df.reindex(columns=self.feature_columns, fill_value=0)

        return self.model.predict(df)

    def prevedi_csv(self,passengersId, dataframe):

        df = pd.get_dummies(dataframe)

        # riallineamento colonne come nel training
        df = df.reindex(columns=self.feature_columns, fill_value=0)

        predizioni = self.model.predict(df)

        output = pd.DataFrame({
            "PassengerId": passengersId,
            "Survived": predizioni
        })

        lista_coppie = list(output.itertuples(index=False, name=None))

        # salva CSV
        output.to_csv("csvs/regLog_prev_test.csv", index=False)

        return lista_coppie
    
