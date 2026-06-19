# Trennklar

Trennklar ist ein mobiler Streamlit-Prototyp zur KI-gestützten Mülltrennung für
Bürgerinnen und Bürger in Duisburg.

## Funktionen

- Foto direkt mit der Kamera aufnehmen oder hochladen
- Gegenstand und Material mit einem multimodalen Modell erkennen
- Entsorgung über ein transparentes lokales Regelwerk empfehlen
- Unsichere und gefährliche Fälle deutlich kennzeichnen
- Vollständiger Demo-Modus ohne API-Schlüssel
- Keine dauerhafte Speicherung hochgeladener Bilder

## App starten

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Danach ist die App normalerweise unter `http://localhost:8501` erreichbar.

## Echte Bilderkennung aktivieren

Für die lokale Entwicklung wird der API-Schlüssel als Umgebungsvariable gesetzt:

```powershell
$env:OPENAI_API_KEY="..."
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Optional lässt sich das Modell über `OPENAI_MODEL` ändern. Ohne Schlüssel bleibt
der Demo-Modus vollständig nutzbar.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Kaggle dataset

The Kaggle dataset should stay local and must not be uploaded to GitHub. Put or
extract it into `data/archive (1)/`. The repository ignores `data/` by default.

Train the free local image classifier:

```powershell
.\.venv\Scripts\python.exe train_local_model.py
```

This creates `models/waste_classifier.json`. Upload that model file to GitHub if
you want Streamlit Cloud to run image recognition without an OpenAI API key.

Create a local manifest from the dataset:

```powershell
.\.venv\Scripts\python.exe evaluate_dataset.py
```

If `kagglehub` is installed and the local dataset is missing, the script can also
download `phenomsg/waste-classification` automatically. With `OPENAI_API_KEY`
set, a small paid API-based sample evaluation can be run:

```powershell
.\.venv\Scripts\python.exe evaluate_dataset.py --sample 20
```

## Grenzen

Der Prototyp ersetzt keine verbindliche kommunale Auskunft. Vor einem Pilottest
sollten alle Regeln mit den Wirtschaftsbetrieben Duisburg abgestimmt und mit
einem repräsentativen Bilddatensatz evaluiert werden.
