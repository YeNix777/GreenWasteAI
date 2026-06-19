from __future__ import annotations

import html
import os

import streamlit as st

from cnn_waste_model import (
    CNN_LABELS_PATH,
    CNN_MODEL_PATH,
    analyze_with_cnn,
    load_cnn_model,
)
from local_waste_model import analyze_with_local_model, load_model
from wastewise import (
    DEFAULT_MODEL,
    DEMO_ITEMS,
    Recognition,
    analyze_image,
    disposal_advice,
)


st.set_page_config(
    page_title="Trennklar",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --green: #176b52;
        --dark: #17362d;
        --cream: #f4f3eb;
        --line: #dfe5df;
    }
    .stApp {
        background:
            radial-gradient(circle at 90% 0%, rgba(64, 145, 110, .14), transparent 28rem),
            var(--cream);
    }
    .block-container { max-width: 760px; padding-top: 1.5rem; padding-bottom: 4rem; }
    h1, h2, h3 { color: var(--dark); letter-spacing: -.025em; }
    [data-testid="stHeader"] { background: transparent; }
    .hero {
        padding: 1.7rem 1.8rem;
        border-radius: 24px;
        color: white;
        background: linear-gradient(135deg, #123c30, #1e755a);
        box-shadow: 0 14px 35px rgba(22, 67, 54, .17);
        margin-bottom: 1.2rem;
    }
    .hero-kicker { color: #bce5d3; font-size: .76rem; font-weight: 800; letter-spacing: .14em; }
    .hero h1 { color: white; font-size: 2.4rem; margin: .25rem 0 .55rem; }
    .hero p { color: #e7f4ee; font-size: 1.04rem; margin: 0; max-width: 580px; }
    .step {
        display: inline-flex; align-items: center; gap: .45rem;
        background: #e1eee8; color: #245844; padding: .42rem .7rem;
        border-radius: 999px; font-size: .78rem; font-weight: 750; margin: .7rem 0;
    }
    .result {
        background: white; border: 1px solid var(--line); border-radius: 22px;
        padding: 1.25rem; box-shadow: 0 8px 24px rgba(24, 54, 45, .07);
    }
    .bin {
        display: flex; gap: 1rem; align-items: center; border-radius: 17px;
        padding: 1rem; margin: .8rem 0; color: white;
    }
    .bin-icon {
        min-width: 48px; height: 48px; border-radius: 14px;
        display: grid; place-items: center; background: rgba(255,255,255,.2);
        font-size: 1rem; font-weight: 900;
    }
    .bin-label { opacity: .83; font-size: .74rem; font-weight: 750; text-transform: uppercase; }
    .bin-name { font-size: 1.35rem; font-weight: 800; }
    .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .7rem; margin: .8rem 0; }
    .detail {
        background: #f5f7f5; border-radius: 13px; padding: .75rem;
        color: #456057; font-size: .84rem;
    }
    .detail strong { color: #1e4236; display: block; font-size: .72rem; text-transform: uppercase; }
    .confidence { height: 8px; background: #e4e9e6; border-radius: 99px; overflow: hidden; }
    .confidence span { display: block; height: 100%; background: #29966e; border-radius: 99px; }
    .note { border-left: 4px solid #df9234; background: #fff6e9; padding: .8rem 1rem; border-radius: 8px; }
    .privacy { color: #607269; font-size: .78rem; text-align: center; margin-top: 1.2rem; }
    @media (max-width: 600px) {
        .block-container { padding: .8rem .8rem 3rem; }
        .hero { border-radius: 18px; padding: 1.35rem; }
        .hero h1 { font-size: 2rem; }
        .detail-grid { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def secret_value(name: str) -> str:
    try:
        return str(st.secrets.get(name, "")).strip()
    except FileNotFoundError:
        return ""


@st.cache_resource(show_spinner=False)
def cached_cnn_model():
    return load_cnn_model()


def render_result(recognition: Recognition) -> None:
    advice = disposal_advice(recognition)
    confidence_percent = round(recognition.confidence * 100)
    st.markdown('<div class="step">3 · Empfehlung</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="result">
            <div style="color:#597068;font-size:.78rem;font-weight:800;text-transform:uppercase">
                Erkannt
            </div>
            <h2 style="margin:.15rem 0">{html.escape(recognition.item)}</h2>
            <div class="detail-grid">
                <div class="detail"><strong>Material</strong>{html.escape(recognition.material)}</div>
                <div class="detail"><strong>Zustand</strong>{html.escape(recognition.condition)}</div>
            </div>
            <div class="bin" style="background:{advice.color}">
                <div class="bin-icon">{advice.icon}</div>
                <div><div class="bin-label">Das gehört zur</div>
                <div class="bin-name">{html.escape(advice.bin_name)}</div></div>
            </div>
            <p><strong>So entsorgst du es:</strong><br>{html.escape(advice.instruction)}</p>
            <p><strong>Vorbereitung:</strong><br>{html.escape(advice.preparation)}</p>
            <div style="display:flex;justify-content:space-between;font-size:.78rem;margin-bottom:.35rem">
                <span>Erkennungssicherheit</span><strong>{confidence_percent}%</strong>
            </div>
            <div class="confidence"><span style="width:{confidence_percent}%"></span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if recognition.confidence < 0.7:
        st.warning(
            "Die Erkennung ist unsicher. Bitte Ergebnis prüfen, Gegenstand einzeln "
            "fotografieren oder ein neues Foto bei gutem Licht aufnehmen."
        )
    if advice.warning:
        st.error(advice.warning)
    with st.expander("Warum diese Empfehlung?"):
        st.write(recognition.explanation)
        st.caption(
            "Die KI erkennt Gegenstand und Material. Die Entsorgung wird anschließend "
            "durch das lokale Regelwerk bestimmt."
        )
    feedback = st.radio(
        "War die Empfehlung hilfreich?",
        ["Noch nicht bewertet", "Ja", "Nein"],
        horizontal=True,
        key=f"feedback_{recognition.item}",
    )
    if feedback == "Ja":
        st.success("Danke für dein Feedback.")
    elif feedback == "Nein":
        st.info("Danke. Im nächsten Entwicklungsschritt wird diese Korrektur gespeichert.")


st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">MÜLL TRENNEN. EINFACH ERKLÄRT.</div>
        <h1>Trennklar</h1>
        <p>Fotografiere deinen Abfall und erhalte eine verständliche
        Entsorgungsempfehlung für Duisburg.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

api_key = secret_value("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "").strip()
model = secret_value("OPENAI_MODEL") or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
cnn_available = CNN_MODEL_PATH.exists() and CNN_LABELS_PATH.exists()
local_model = load_model()

with st.sidebar:
    st.header("Über den Prototyp")
    st.write(
        "Trennklar ist ein Studienprototyp. Kommunale Regeln können sich ändern; "
        "verbindlich ist das Abfall-ABC der Wirtschaftsbetriebe Duisburg."
    )
    st.link_button(
        "Duisburger Abfall-ABC öffnen",
        "https://www.wb-duisburg.de/unsere-leistungen/abfall-und-wertstoffe/abfall-abc",
        width="stretch",
    )
    st.divider()
    if api_key:
        st.caption(f"Bildmodell: {model}")
    elif cnn_available:
        st.caption("Bildmodell: MobileNetV2-CNN, lokal trainiert")
    elif local_model:
        st.caption("Bildmodell: einfache lokale Baseline")
    else:
        st.caption("Bild-KI: nicht konfiguriert")

mode = st.segmented_control(
    "Modus",
    ["Foto analysieren", "Demo ausprobieren"],
    default="Foto analysieren",
    label_visibility="collapsed",
)

recognition = None

if mode == "Foto analysieren":
    st.markdown('<div class="step">1 · Foto hinzufügen</div>', unsafe_allow_html=True)
    source = st.radio(
        "Bildquelle",
        ["Kamera", "Datei hochladen"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if source == "Kamera":
        image = st.camera_input(
            "Gegenstand fotografieren",
            help="Am besten nur einen Gegenstand bei gutem Licht aufnehmen.",
        )
    else:
        image = st.file_uploader(
            "Foto auswählen",
            type=["jpg", "jpeg", "png", "webp"],
            help="Unterstützt werden JPG, PNG und WebP.",
        )

    st.markdown('<div class="step">2 · Bild prüfen</div>', unsafe_allow_html=True)
    if image is None:
        st.info("Fotografiere einen einzelnen Gegenstand oder lade ein Bild hoch.")
    else:
        image_bytes = image.getvalue()
        image_id = hash(image_bytes)
        if st.session_state.get("last_image_id") != image_id:
            st.session_state.pop("last_recognition", None)
        st.image(image, caption="Dieses Bild wird analysiert.", width="stretch")
        if not api_key and not cnn_available and not local_model:
            st.warning(
                "Die Bild-KI ist noch nicht eingerichtet. Trainiere das lokale "
                "Modell oder nutze den Demo-Modus."
            )
        if st.button(
            "Abfall erkennen",
            type="primary",
            width="stretch",
            disabled=not api_key and not cnn_available and not local_model,
        ):
            with st.spinner("Gegenstand und Material werden erkannt …"):
                try:
                    if api_key:
                        recognition = analyze_image(
                            image_bytes,
                            image.type or "image/jpeg",
                            api_key,
                            model,
                        )
                    elif cnn_available:
                        cnn_model = cached_cnn_model()
                        recognition = analyze_with_cnn(image_bytes, cnn_model)
                    else:
                        recognition = analyze_with_local_model(image_bytes, local_model)
                    st.session_state["last_recognition"] = recognition
                    st.session_state["last_image_id"] = image_id
                except (RuntimeError, ValueError) as exc:
                    st.error(f"Analyse fehlgeschlagen: {exc}")
        recognition = recognition or st.session_state.get("last_recognition")
else:
    st.markdown('<div class="step">1 · Beispiel wählen</div>', unsafe_allow_html=True)
    demo_name = st.selectbox("Welchen Gegenstand möchtest du testen?", DEMO_ITEMS)
    st.caption("Der Demo-Modus zeigt den vollständigen Ablauf ohne Bild-API.")
    if st.button("Beispiel auswerten", type="primary", width="stretch"):
        recognition = DEMO_ITEMS[demo_name]

if recognition:
    render_result(recognition)

st.markdown(
    """
    <div class="privacy">
        🔒 Hochgeladene Bilder werden von dieser App nicht dauerhaft gespeichert.<br>
        Prototyp · Region Duisburg · Empfehlungen ohne Gewähr
    </div>
    """,
    unsafe_allow_html=True,
)
