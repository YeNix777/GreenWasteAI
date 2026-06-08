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

## Grenzen

Der Prototyp ersetzt keine verbindliche kommunale Auskunft. Vor einem Pilottest
sollten alle Regeln mit den Wirtschaftsbetrieben Duisburg abgestimmt und mit
einem repräsentativen Bilddatensatz evaluiert werden.
