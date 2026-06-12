from io import BytesIO
import base64

import pandas as pd
from flask import Flask, jsonify, request, render_template_string
import matplotlib.pyplot as plt

from it.Brialemi_SRL.dataset.dataset_manager import DatasetManager
from it.Brialemi_SRL.machine_learning.regressione_logistica import RegLogistica
from it.Brialemi_SRL.machine_learning.random_forest import RndForest
from it.Brialemi_SRL.machine_learning.xg_boost import XgBoost

class FlaskManager(object): # è una classe INTERFACCIA
    def __init__(self):
        self.app = Flask(__name__)
        self.__register_routes() # inizializzazione delle varie root tutte insieme

        self.ds_mg = DatasetManager()

        self.ds_mg.clean()
        self.reg_log = RegLogistica(self.ds_mg.get_datatrain())
        self.rnd_forest = RndForest(self.ds_mg.get_datatrain())
        self.xgb = XgBoost(self.ds_mg.get_datatrain())
        # in questo modo sia il cleaning sia i modelli vengono stimati in automatico

    def run(self, **kwargs):
        self.app.run(**kwargs) #**kwargs non specifico i parametri che dopo dovrò inserire

    def __register_routes(self): # __ indica un metodo che vede solo questa classe
        @self.app.route('/') # in automatico il metodo è GET 
        def home():
            return "Flask online"

        @self.app.route('/datasetshow')
        def dataset_show():
            return jsonify(self.ds_mg.get_datatrain().head(5).to_dict()) # per ritornare un dataframe si usa .to_dict()

        @self.app.route('/info')
        def info():
            risp = self.ds_mg.analisi()

            def convert(obj):
                if isinstance(obj, pd.Series):
                    return obj.to_dict()
                return obj

            risp = {k: convert(v) for k, v in risp.items()}

            return jsonify(risp)

        @self.app.route('/grafici')
        def grafici():
            grafici = self.ds_mg.grafici()
            figs = []
            for value in grafici.values(): # bisogna scomporre la scritta dall'immagine
                if value is None:
                    continue

                # Se è una lista di figure
                if isinstance(value, list): # infatti gli istogrammi sono più immagini in una lista
                    figs.extend(value) 
                else:
                    figs.append(value)

            images = []

            for fig in figs:
                img = BytesIO()
                fig.savefig(img, format="png", bbox_inches="tight")
                img.seek(0)

                encoded = base64.b64encode(img.getvalue()).decode()
                images.append(encoded)

                plt.close(fig)

            html = """
               <h1>Plots</h1>
               {% for img in images %}
                   <img src="data:image/png;base64,{{ img }}" style="margin:10px;">
               {% endfor %}
               """

            return render_template_string(html, images=images) #prende il template html e usa su tutte le immagini 
                                                               #trasformate in formato base64

        @self.app.route('/correlazione')
        def correlazione():
            return jsonify(self.ds_mg.correlazione())

        @self.app.route('/valMod_regLogistica')
        def valMod_regLogistica():
            return jsonify(self.reg_log.get_val())

        @self.app.route('/valMod_rndForest')
        def valMod_regLogisticaRndForest():
            return jsonify(self.rnd_forest.get_val())

        @self.app.route('/valMod_XGboost')
        def valMod_XGboost():
            return jsonify(self.xgb.get_val())

        @self.app.route('/confronto_valutazioni')
        def confronto_valutazioni():
            modelli = {
                "regressione_logistica": self.reg_log.get_val(),
                "random_forest": self.rnd_forest.get_val(),
                "xgboost": self.xgb.get_val()
            }
            # tabella principale (confronto accuracy)
            tabella_accuracy = [
                {
                    "modello": nome,
                    "accuracy": valori["accuracy"]
                }
                for nome, valori in modelli.items()
            ]

            # ordinamento per performance
            tabella_accuracy = sorted(
                tabella_accuracy,
                key=lambda x: x["accuracy"],
                reverse=True
            )

            migliore = tabella_accuracy[0]["modello"]

            return jsonify({
                "tabella_accuracy": tabella_accuracy,
                "migliore": migliore,
                "dettaglio_modelli": modelli
            })

        @self.app.route('/previsione_regLogistica', methods=['POST'])
        def previsione_regLogistica():
            data = request.get_json() 
            obj = [
                ('Pclass', data.get('Pclass')),
                ('Sex', data.get('Sex')),
                ('Age', data.get('Age')),
                ('SibSp', data.get('SibSp')),
                ('Parch', data.get('Parch')),
                ('Fare', data.get('Fare')),
                ('Embarked', data.get('Embarked'))
            ]   # creando questo dizionario posso controllare che siano presenti tutti gli attributi necessari
             
            pred = self.reg_log.prevedi(obj)
            print("PREDIZIONE:", pred)

            return jsonify({"survived status": self.reg_log.prevedi(obj).tolist()})

        @self.app.route('/previsione_rndForest', methods=['POST'])
        def previsione_rndForest():
            data = request.get_json()
            obj = [
                ('Pclass', data.get('Pclass')),
                ('Sex', data.get('Sex')),
                ('Age', data.get('Age')),
                ('SibSp', data.get('SibSp')),
                ('Parch', data.get('Parch')),
                ('Fare', data.get('Fare')),
                ('Embarked', data.get('Embarked'))
            ]
            return jsonify({"survived status": self.rnd_forest.prevedi(obj).tolist()})

        @self.app.route('/previsione_xgboost', methods=['POST'])
        def previsione_xgboost():
            data = request.get_json()
            obj = [
                ('Pclass', data.get('Pclass')),
                ('Sex', data.get('Sex')),
                ('Age', data.get('Age')),
                ('SibSp', data.get('SibSp')),
                ('Parch', data.get('Parch')),
                ('Fare', data.get('Fare')),
                ('Embarked', data.get('Embarked'))
            ]
            return jsonify({"survived status": self.xgb.prevedi(obj).tolist()})

        @self.app.route('/previsione_regLogistica_test')
        def previsione_regLosgistica_test():
            passId = self.ds_mg.get_datatest()["PassengerId"].copy()
            cleanDf = self.ds_mg.clean_data(self.ds_mg.get_datatest())
            ris = self.reg_log.prevedi_csv(passId, cleanDf)
            return jsonify({"previsioni del dataset: ": ris})

        @self.app.route('/previsione_rndForest_test')
        def previsione_rndForest_test():
            passId = self.ds_mg.get_datatest()["PassengerId"].copy()
            cleanDf = self.ds_mg.clean_data(self.ds_mg.get_datatest())
            ris = self.rnd_forest.prevedi_csv(passId, cleanDf)
            return jsonify({"previsioni del dataset: ": ris})

        @self.app.route('/previsione_xgboost_test')
        def previsione_xgboost_test():
            passId = self.ds_mg.get_datatest()["PassengerId"].copy()
            cleanDf = self.ds_mg.clean_data(self.ds_mg.get_datatest())
            ris = self.xgb.prevedi_csv(passId, cleanDf)
            return jsonify({"previsioni del dataset: ": ris})