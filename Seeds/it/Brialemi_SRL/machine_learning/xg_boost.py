import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

class XgBoost:

    def __init__(self, dataset):
        self.val_model = None
        self.set_mod(dataset)

    def set_mod(self, dataset):
        # Variabili esplicative e target
        y = dataset["Survived"]

        # feature categoriche
        X = pd.get_dummies(
            dataset.drop(columns=["Survived"]),
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
        # Modello XGBoost
        model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.3,
            max_depth=5,
            random_state=42,
            eval_metric="logloss",
        )

        # Addestramento
        model.fit(X_train, y_train)

        self.model = model
        y_pred = model.predict(X_test)

        importance_df = pd.DataFrame({
            "variabile": X.columns,
            "feature_importance": model.feature_importances_
        })
        importance_df["abs_importance"] = importance_df["feature_importance"].abs()
        importance_df = importance_df.sort_values(
            by="feature_importance",
            ascending=False
        )

        # Valutazione
        self.val_model = {
            "predizioni": y_pred.tolist(),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            # opzionali ma utili:
            "feature_importance": importance_df[["variabile", "feature_importance"]].to_dict("records")
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

        return self.model.predict(df)[0]

    def prevedi_csv(self, passengerId, dataframe):
        df = pd.get_dummies(dataframe)

        # riallineamento colonne come nel training
        df = df.reindex(columns=self.feature_columns, fill_value=0)

        predizioni = self.model.predict(df)

        output = pd.DataFrame({
            "PassengerId": passengerId,
            "Survived": predizioni
        })

        lista_coppie = list(output.itertuples(index=False, name=None))

        # salva CSV
        output.to_csv("csvs/xgboost_prev_test.csv", index=False)

        return lista_coppie