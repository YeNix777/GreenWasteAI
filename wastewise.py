from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass


DEFAULT_MODEL = "gpt-5.4-mini"


@dataclass(frozen=True)
class Recognition:
    item: str
    material: str
    category: str
    condition: str
    hazardous: bool
    confidence: float
    explanation: str


@dataclass(frozen=True)
class DisposalAdvice:
    bin_name: str
    color: str
    icon: str
    instruction: str
    preparation: str
    warning: str = ""


DEMO_ITEMS = {
    "Joghurtbecher": Recognition(
        "Joghurtbecher",
        "Kunststoffverpackung",
        "packaging",
        "restentleert",
        False,
        0.96,
        "Ein Joghurtbecher ist eine Verkaufsverpackung aus Kunststoff.",
    ),
    "Bananenschale": Recognition(
        "Bananenschale",
        "organisches Material",
        "organic",
        "unverpackt",
        False,
        0.98,
        "Die Schale ist ein kompostierbarer Küchenabfall.",
    ),
    "Pizzakarton (fettig)": Recognition(
        "stark verschmutzter Pizzakarton",
        "beschmutzte Pappe",
        "residual",
        "fettig und verschmutzt",
        False,
        0.91,
        "Stark verschmutzte Pappe lässt sich nicht sinnvoll als Altpapier recyceln.",
    ),
    "Zeitung": Recognition(
        "Zeitung",
        "Papier",
        "paper",
        "sauber und trocken",
        False,
        0.99,
        "Sauberes Papier kann über die Papiersammlung recycelt werden.",
    ),
    "Leere Glasflasche": Recognition(
        "leere Glasflasche",
        "Verpackungsglas",
        "glass",
        "leer",
        False,
        0.98,
        "Leere Glasverpackungen werden nach Farbe im Altglascontainer gesammelt.",
    ),
    "Altes Smartphone": Recognition(
        "Smartphone",
        "Elektronik mit Akku",
        "electronics",
        "ausgedient",
        True,
        0.97,
        "Das Gerät enthält Elektronik und einen Akku und gehört nie in den Hausmüll.",
    ),
    "Batterie": Recognition(
        "Haushaltsbatterie",
        "Batterie",
        "battery",
        "gebraucht",
        True,
        0.99,
        "Batterien können Schadstoffe enthalten und bei falscher Entsorgung Brände auslösen.",
    ),
}


def clamp_confidence(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def recognition_from_dict(data: dict) -> Recognition:
    return Recognition(
        item=str(data.get("item", "Unbekannter Gegenstand")).strip(),
        material=str(data.get("material", "nicht sicher erkannt")).strip(),
        category=str(data.get("category", "unknown")).strip().lower(),
        condition=str(data.get("condition", "nicht sicher erkannt")).strip(),
        hazardous=bool(data.get("hazardous", False)),
        confidence=clamp_confidence(data.get("confidence", 0)),
        explanation=str(data.get("explanation", "")).strip(),
    )


def disposal_advice(recognition: Recognition) -> DisposalAdvice:
    category = recognition.category

    if category == "battery":
        return DisposalAdvice(
            "Batteriesammlung",
            "#dc5a47",
            "BAT",
            "Kostenlos im Handel oder an einer kommunalen Sammelstelle abgeben.",
            "Pole loser Lithium-Akkus möglichst abkleben. Beschädigte Akkus getrennt und vorsichtig transportieren.",
            "Batterien und Akkus niemals in Restmüll oder Wertstofftonne werfen: Brandgefahr.",
        )
    if category == "electronics":
        return DisposalAdvice(
            "Elektroaltgeräte-Sammlung",
            "#7657a8",
            "E",
            "Beim Handel zurückgeben oder zu einem Duisburger Recyclinghof bringen.",
            "Entnehmbare Batterien und Akkus vorher entfernen und separat sammeln.",
            "Elektrogeräte gehören nicht in den Hausmüll.",
        )
    if category == "hazardous" or recognition.hazardous:
        return DisposalAdvice(
            "Schadstoffsammlung",
            "#b6413b",
            "!",
            "Zur kommunalen Schadstoffsammlung oder zum dafür vorgesehenen Recyclinghof bringen.",
            "Im Originalbehälter lassen, sicher verschließen und Stoffe nicht vermischen.",
            "Nicht in Ausguss, Toilette oder Hausmüll geben.",
        )

    advice = {
        "packaging": DisposalAdvice(
            "Wertstofftonne",
            "#f2c94c",
            "W",
            "In Duisburg in die gelbe Wertstofftonne geben.",
            "Restentleeren genügt. Bestandteile wie Deckel und Becher voneinander trennen; Ausspülen ist nicht nötig.",
        ),
        "paper": DisposalAdvice(
            "Papiertonne",
            "#4f86c6",
            "P",
            "In die blaue Papiertonne oder einen Papiercontainer geben.",
            "Nur sauber und trocken einwerfen. Kunststofffolien vorher entfernen.",
        ),
        "glass": DisposalAdvice(
            "Altglascontainer",
            "#4f9b70",
            "G",
            "Nach Weiß-, Braun- oder Grünglas sortiert in den Altglascontainer geben.",
            "Deckel vorher abnehmen. Trinkgläser, Keramik und Fensterglas gehören nicht hinein.",
        ),
        "organic": DisposalAdvice(
            "Biotonne",
            "#8b6846",
            "B",
            "Ohne Kunststoffbeutel in die braune Biotonne geben.",
            "Lose, in Zeitungspapier oder in einer Papiertüte entsorgen.",
        ),
        "residual": DisposalAdvice(
            "Restmüll",
            "#59636e",
            "R",
            "In die graue Restmülltonne geben.",
            "Nur nicht verwertbare und ungefährliche Haushaltsabfälle einwerfen.",
        ),
        "textile": DisposalAdvice(
            "Altkleidersammlung",
            "#3c8d8d",
            "T",
            "Sauber und trocken über eine seriöse Altkleidersammlung abgeben.",
            "Stark verschmutzte oder nasse Textilien gehören in den Restmüll.",
        ),
    }
    return advice.get(
        category,
        DisposalAdvice(
            "Noch nicht eindeutig",
            "#d9843b",
            "?",
            "Bitte das kommunale Abfall-ABC prüfen oder ein deutlicheres Foto aufnehmen.",
            "Material, Verschmutzung und mögliche Batterien kontrollieren.",
            "Bei Unsicherheit nicht in eine Wertstoffsammlung geben.",
        ),
    )


def _extract_json(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("Die Bilderkennung hat kein lesbares Ergebnis geliefert.")
    return json.loads(match.group(0))


def _response_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    parts = []
    for output in payload.get("output", []):
        for content in output.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    if not parts:
        raise ValueError("Die API-Antwort enthielt keinen Text.")
    return "\n".join(parts)


def analyze_image(
    image_bytes: bytes,
    mime_type: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> Recognition:
    if not api_key:
        raise ValueError("Für die echte Bilderkennung fehlt der API-Schlüssel.")
    if not image_bytes:
        raise ValueError("Es wurde kein Bild übergeben.")

    image_data = base64.b64encode(image_bytes).decode("ascii")
    prompt = """
Du analysierst genau einen Haushaltsabfall für eine Mülltrennungs-App in Duisburg.
Erkenne den sichtbaren Hauptgegenstand und sein Material. Antworte ausschließlich
als einzelnes JSON-Objekt mit diesen Feldern:
item (deutsch), material (deutsch), category, condition (deutsch), hazardous
(boolean), confidence (0 bis 1), explanation (ein kurzer deutscher Satz).

category muss exakt einer dieser Werte sein:
packaging, paper, glass, organic, residual, electronics, battery, hazardous,
textile, unknown.

Wichtig: "packaging" nur für Verpackungen oder Duisburger Wertstoffe aus Kunststoff
oder Metall. Stark verschmutztes Papier ist residual. Geräte mit Elektronik sind
electronics. Batterien sind battery. Bei unklarem Bild nutze unknown und eine
niedrige confidence. Erfinde keine nicht sichtbaren Details.
""".strip()
    body = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{image_data}",
                        "detail": "low",
                    },
                ],
            }
        ],
        "max_output_tokens": 350,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(details)["error"]["message"]
        except (KeyError, TypeError, json.JSONDecodeError):
            message = f"API-Fehler {exc.code}"
        raise RuntimeError(message) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Die Bilderkennung ist derzeit nicht erreichbar.") from exc

    return recognition_from_dict(_extract_json(_response_text(payload)))


def configured_api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()
