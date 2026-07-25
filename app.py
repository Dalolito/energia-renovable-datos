import os

import streamlit as st
from groq import Groq

# ------------------------------------------------------------------
# Configuración general de la página
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Bot de Cultura General e Historia Mundial",
    page_icon="🏛️",
    layout="centered",
)

MODELO = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "Eres un asistente experto en cultura general e historia mundial. "
    "Respondes de forma clara, precisa y educativa. Cuando sea relevante, "
    "das contexto histórico (fechas, lugares, personajes clave) y "
    "aclaras si un dato es incierto o debatido entre historiadores. "
    "Si te preguntan algo fuera de cultura general o historia, respondes "
    "brevemente y sugieres reformular hacia esos temas."
)

# ------------------------------------------------------------------
# Obtener API key: variable de entorno / st.secrets (recomendado)
# ------------------------------------------------------------------
def obtener_api_key():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = None
    return api_key


api_key = obtener_api_key()

with st.sidebar:
    st.header("⚙️ Configuración")

    api_key_manual = st.text_input(
        "🔑 Groq API Key",
        value=api_key if api_key else "",
        type="password",
        placeholder="gsk_...",
        help="Pega aquí tu API Key de Groq. También puedes configurarla como variable de entorno o en st.secrets.",
    )
    if api_key_manual:
        api_key = api_key_manual

    if api_key:
        st.success("API Key cargada ✅")
    else:
        st.warning("Falta la API Key para poder chatear.")

    st.divider()
    temperatura = st.slider("Creatividad (temperature)", 0.0, 1.0, 0.5, 0.1)
    max_tokens = st.slider("Longitud máxima de respuesta (tokens)", 256, 2048, 1024, 128)

    st.divider()
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.mensajes = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.rerun()

if not api_key:
    st.warning("Ingresa tu API Key de Groq en la barra lateral para comenzar.")
    st.stop()

client = Groq(api_key=api_key)

# ------------------------------------------------------------------
# Estado de la conversación
# ------------------------------------------------------------------
if "mensajes" not in st.session_state:
    st.session_state.mensajes = [{"role": "system", "content": SYSTEM_PROMPT}]

# ------------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------------
st.title("🏛️ Bot de Cultura General e Historia Mundial")
st.caption(f"Powered by Groq · Modelo: {MODELO}")

# ------------------------------------------------------------------
# Mostrar historial (sin el mensaje "system")
# ------------------------------------------------------------------
for mensaje in st.session_state.mensajes:
    if mensaje["role"] == "system":
        continue
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])

# ------------------------------------------------------------------
# Input del usuario
# ------------------------------------------------------------------
pregunta = st.chat_input("Pregúntame sobre historia mundial o cultura general...")

if pregunta:
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        respuesta_completa = ""
        try:
            stream = client.chat.completions.create(
                model=MODELO,
                messages=st.session_state.mensajes,
                temperature=temperatura,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                respuesta_completa += delta
                placeholder.markdown(respuesta_completa + "▌")
            placeholder.markdown(respuesta_completa)
        except Exception as e:
            respuesta_completa = f"⚠️ Ocurrió un error al llamar a la API de Groq: {e}"
            placeholder.markdown(respuesta_completa)

    st.session_state.mensajes.append({"role": "assistant", "content": respuesta_completa})

# ------------------------------------------------------------------
# Sugerencias rápidas
# ------------------------------------------------------------------
if len(st.session_state.mensajes) == 1:
    st.markdown("**Prueba preguntando:**")
    ejemplos = [
        "¿Cuáles fueron las causas de la Primera Guerra Mundial?",
        "¿Quién fue Cleopatra y por qué es tan famosa?",
        "Explícame la caída del Imperio Romano",
        "¿Qué fue la Guerra Fría?",
    ]
    cols = st.columns(2)
    for i, ejemplo in enumerate(ejemplos):
        if cols[i % 2].button(ejemplo):
            st.session_state.mensajes.append({"role": "user", "content": ejemplo})
            st.rerun()
