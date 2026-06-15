# KMeans Seeds API

API Flask per analizzare il dataset **Seeds** e applicare un modello di clustering **K-Means**. Il progetto carica il dataset locale, esegue una fase di pulizia e analisi esplorativa, addestra un modello KMeans con 3 cluster e rende disponibili risultati, grafici e predizioni tramite endpoint HTTP.

L'applicazione e' disponibile anche su Docker Hub come:

```bash
asallemi/kmeans
```

## Obiettivo del progetto

Il progetto ha lo scopo di raggruppare campioni di semi in cluster usando caratteristiche morfologiche come area, perimetro, compattezza, lunghezza e larghezza del kernel, asimmetria e lunghezza del solco. La colonna `classe` viene usata per valutare la coerenza dei cluster, ma non viene utilizzata come feature di addestramento del modello KMeans.

## Dataset

Il dataset si trova in:

```text
Seeds/seeds_dataset.txt
```

Le colonne caricate dal progetto sono:

| Colonna | Descrizione |
| --- | --- |
| `area` | Area del seme |
| `perimetro` | Perimetro del seme |
| `compattezza` | Compattezza |
| `lunghezza_kernel` | Lunghezza del kernel |
| `larghezza_kernel` | Larghezza del kernel |
| `asimmetria` | Coefficiente di asimmetria |
| `lunghezza_solco` | Lunghezza del solco |
| `classe` | Classe reale del seme |

## Funzionalita'

- Caricamento automatico del dataset Seeds.
- Pulizia dei dati e gestione dei valori mancanti.
- Analisi esplorativa con statistiche, outlier, test di normalita' e PCA.
- Grafici di correlazione, istogrammi, PCA ed elbow method.
- Addestramento automatico di un modello KMeans.
- Valutazione del clustering con `silhouette_score`, `rand_index`, `inertia` e centroidi.
- API Flask per consultare dataset, metriche, grafici e predizioni.
- Esecuzione tramite Docker o ambiente Python locale.

## Struttura del progetto

```text
Kmeans/
+-- Dockerfile
+-- README.md
+-- main.py
+-- requirements.txt
+-- Seeds/
    +-- seeds_dataset.txt
    +-- it/
        +-- Brialemi_SRL/
            +-- dataset/
            |   +-- dataset_analisi.py
            |   +-- dataset_manager.py
            |   +-- grafici.py
            +-- flask/
            |   +-- flask_manager.py
            +-- machine_learning/
                +-- kmeans.py
                +-- random_forest.py
                +-- regressione_logistica.py
                +-- xg_boost.py
```

## Tecnologie utilizzate

- Python 3.11
- Flask
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Docker

## Avvio con Docker Hub

Scarica ed esegui direttamente l'immagine pubblicata:

```bash
docker pull asallemi/kmeans
docker run -p 5000:5000 asallemi/kmeans
```

L'app sara' disponibile su:

```text
http://localhost:5000
```

## Avvio locale

Clona il repository:

```bash
git clone <url-del-repository>
cd Kmeans
```

Crea e attiva un ambiente virtuale:

```bash
python -m venv .venv
```

Su Windows:

```bash
.venv\Scripts\activate
```

Su macOS/Linux:

```bash
source .venv/bin/activate
```

Installa le dipendenze:

```bash
pip install -r requirements.txt
```

Avvia l'applicazione:

```bash
python main.py
```

Il server Flask parte su:

```text
http://localhost:5000
```

## Build Docker locale

In alternativa puoi costruire l'immagine localmente:

```bash
docker build -t kmeans-seeds .
docker run -p 5000:5000 kmeans-seeds
```

## Endpoint API

### Home

```http
GET /
```

Restituisce informazioni sul servizio e la lista degli endpoint disponibili.

### Anteprima dataset

```http
GET /datasetshow
```

Restituisce le prime righe del dataset dopo il caricamento.

### Analisi dataset

```http
GET /info
```

Restituisce informazioni di analisi esplorativa, tra cui valori nulli, valori anomali, outlier, test di normalita' e PCA.

### Grafici esplorativi

```http
GET /grafici
```

Restituisce una pagina HTML con grafici generati da Matplotlib, tra cui correlazione, istogrammi, PCA ed elbow method.

### Correlazione

```http
GET /correlazione
```

Restituisce la matrice di correlazione del dataset.

### Valutazione KMeans

```http
GET /valMod_kmeans
```

Restituisce le metriche del modello KMeans:

- `silhouette_score`
- `rand_index`
- `inertia`
- `centroidi`

### Predizione cluster

```http
POST /prevedi_kmeans
```

Esempio di payload JSON:

```json
{
  "osservazione": {
    "area": 15.26,
    "perimetro": 14.84,
    "compattezza": 0.871,
    "lunghezza_kernel": 5.763,
    "larghezza_kernel": 3.312,
    "asimmetria": 2.221,
    "lunghezza_solco": 5.22
  }
}
```

Esempio con `curl`:

```bash
curl -X POST http://localhost:5000/prevedi_kmeans \
  -H "Content-Type: application/json" \
  -d "{\"osservazione\":{\"area\":15.26,\"perimetro\":14.84,\"compattezza\":0.871,\"lunghezza_kernel\":5.763,\"larghezza_kernel\":3.312,\"asimmetria\":2.221,\"lunghezza_solco\":5.22}}"
```

### Plot KMeans

```http
GET /plot_kmeans
```

Restituisce una pagina HTML con il grafico dei cluster KMeans nello spazio PCA e i centroidi evidenziati.

## Esempio risposta principale

```json
{
  "service": "Kmeans API",
  "version": "1.0.0",
  "endpoints": {
    "/datasetshow": "GET - Head dataset",
    "/info": "GET - Statistiche descrittive",
    "/grafici": "GET - Grafici di correlazione, distribuzioni e PCA",
    "/correlazione": "GET - Matrice di correlazione",
    "/valMod_kmeans": "GET - Kmeans",
    "/prevedi_kmeans": "POST - Previsioni su file di test",
    "/plot_kmeans": "GET - Plot dei cluster"
  }
}
```

## Note sul modello

Il modello KMeans viene inizializzato con:

- `n_clusters=3`
- `random_state=42`
- `n_init=10`
- PCA a 2 componenti per visualizzazione e clustering

La classe reale del dataset viene esclusa dalle feature di addestramento e usata solo per calcolare il `rand_index`.

## Autore

Progetto sviluppato da **Alisia Sallemi**.
