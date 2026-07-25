import os
import json
import re

import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq

# ------------------------------------------------------------------
# Configuración general
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Entrevistas a Datos + EDA",
    page_icon="🎙️",
    layout="wide",
)

MODELO = "llama-3.3-70b-versatile"

CAMPOS_DEFAULT = [
    {"campo": "nombre", "descripcion": "Nombre de la persona entrevistada (si se menciona, si no null)"},
    {"campo": "edad", "descripcion": "Edad en años (número, si no se menciona: null)"},
    {"campo": "num_hijos", "descripcion": "Número total de hijos (entero)"},
    {"campo": "hijas_mujeres", "descripcion": "Número de hijas mujeres (entero)"},
    {"campo": "hijos_hombres", "descripcion": "Número de hijos hombres (entero)"},
    {"campo": "ocupacion", "descripcion": "Ocupación o profesión mencionada"},
    {"campo": "ciudad", "descripcion": "Ciudad o lugar de residencia mencionado"},
    {"campo": "estado_civil", "descripcion": "Estado civil si se menciona"},
]

# ------------------------------------------------------------------
# API Key
# ------------------------------------------------------------------
def obtener_api_key():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = None
    return api_key


with st.sidebar:
    st.header("🔑 Groq API")
    api_key_env = obtener_api_key()
    api_key = st.text_input(
        "Groq API Key", value=api_key_env if api_key_env else "",
        type="password", placeholder="gsk_...",
    )
    st.success("API Key cargada ✅") if api_key else st.warning("Falta la API Key.")

    st.divider()
    st.header("🧩 Campos a extraer")
    st.caption("Edita el JSON si quieres cambiar qué campos se extraen de cada entrevista.")
    campos_json_texto = st.text_area(
        "Esquema de campos (JSON)",
        value=json.dumps(CAMPOS_DEFAULT, ensure_ascii=False, indent=2),
        height=300,
    )
    try:
        campos_schema = json.loads(campos_json_texto)
    except json.JSONDecodeError:
        st.error("El JSON de campos no es válido. Se usará el esquema por defecto.")
        campos_schema = CAMPOS_DEFAULT

# ------------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------------
st.title("🎙️ Entrevistas a Datos Estructurados + EDA")
st.markdown(
    "Pega un texto con **varias entrevistas juntas** (una tras otra, en el mismo párrafo o "
    "separadas por saltos de línea). El modelo identifica cada entrevista y extrae los campos "
    "definidos en el sidebar, y luego se corre un EDA automático sobre la tabla resultante."
)

texto_entrevistas = st.text_area(
    "Pega aquí el texto con las entrevistas",
    height=250,
    placeholder=(
        "Ej: Entrevistado 1: cuántos hijos tienes... yo tengo dos, una niña y un niño... "
        "Entrevistado 2: yo tengo tres hijos, todos hombres..."
    ),
)

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    procesar = st.button("🚀 Extraer datos", type="primary")

# ------------------------------------------------------------------
# Extracción vía LLM
# ------------------------------------------------------------------
def extraer_json_de_texto(texto: str):
    """Busca el primer bloque JSON válido (array) dentro de un texto."""
    match = re.search(r"\[.*\]", texto, re.DOTALL)
    if not match:
        raise ValueError("El modelo no devolvió un array JSON reconocible.")
    return json.loads(match.group(0))


def construir_prompt(campos, texto):
    lista_campos = "\n".join([f"- {c['campo']}: {c['descripcion']}" for c in campos])
    nombres_campos = [c["campo"] for c in campos]
    return f"""Eres un extractor de datos estructurados. A continuación hay un texto que
contiene VARIAS entrevistas mezcladas en el mismo párrafo o bloque de texto
(pueden no estar claramente separadas). Tu tarea:

1. Identifica cuántas entrevistas/personas distintas hay en el texto.
2. Para cada una, extrae los siguientes campos:
{lista_campos}

Reglas:
- Responde EXCLUSIVAMENTE con un array JSON válido, sin texto adicional, sin explicaciones,
  sin backticks de markdown.
- Cada elemento del array es un objeto con exactamente estas claves: {nombres_campos}
- Si un dato no se menciona explícita o implícitamente, usa null.
- Los campos numéricos deben ser números (int), no strings.
- No inventes información que no esté en el texto.

TEXTO:
\"\"\"
{texto}
\"\"\"

Responde solo con el array JSON.
"""


if procesar:
    if not api_key:
        st.error("Ingresa tu Groq API Key en el sidebar.")
        st.stop()
    if not texto_entrevistas.strip():
        st.error("Pega el texto de las entrevistas primero.")
        st.stop()

    client = Groq(api_key=api_key)
    prompt = construir_prompt(campos_schema, texto_entrevistas)

    with st.spinner("Extrayendo datos estructurados con Llama 3.3 70B..."):
        try:
            respuesta = client.chat.completions.create(
                model=MODELO,
                messages=[
                    {"role": "system", "content": "Eres un extractor de datos estructurados preciso y confiable."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            contenido = respuesta.choices[0].message.content
            registros = extraer_json_de_texto(contenido)
            df_entrevistas = pd.DataFrame(registros)
            st.session_state.df_entrevistas = df_entrevistas
            st.success(f"✅ Se extrajeron {len(df_entrevistas)} entrevistas/registros.")
        except Exception as e:
            st.error(f"Error al extraer datos: {e}")
            st.stop()

# ------------------------------------------------------------------
# Resultado + EDA
# ------------------------------------------------------------------
if "df_entrevistas" in st.session_state and not st.session_state.df_entrevistas.empty:
    df = st.session_state.df_entrevistas.copy()

    st.divider()
    st.subheader("📋 Tabla de Datos Extraídos")
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar tabla como CSV", csv_bytes, "entrevistas_extraidas.csv", "text/csv")

    st.divider()
    st.subheader("🔍 EDA Automático")

    # Separar columnas numéricas y categóricas dinámicamente
    columnas_numericas = df.select_dtypes(include=["number"]).columns.tolist()
    columnas_categoricas = [c for c in df.columns if c not in columnas_numericas]

    # --- Calidad de datos ---
    st.markdown("**Calidad de datos**")
    col_a, col_b = st.columns(2)
    with col_a:
        nulos = df.isnull().sum().reset_index()
        nulos.columns = ["Campo", "Valores Nulos"]
        st.dataframe(nulos, use_container_width=True, hide_index=True)
    with col_b:
        st.metric("Total de registros (entrevistas)", len(df))
        st.metric("Columnas extraídas", len(df.columns))
        st.metric("Filas duplicadas", int(df.duplicated().sum()))

    # --- Estadísticas numéricas ---
    if columnas_numericas:
        st.markdown("**Estadísticas descriptivas (numéricas)**")
        st.dataframe(df[columnas_numericas].describe().round(2), use_container_width=True)

        st.markdown("**Distribuciones numéricas**")
        cols_charts = st.columns(min(3, len(columnas_numericas)))
        for i, col in enumerate(columnas_numericas):
            with cols_charts[i % len(cols_charts)]:
                fig = px.histogram(df, x=col, title=f"Distribución de {col}", nbins=10)
                st.plotly_chart(fig, use_container_width=True)

    # --- Categóricas ---
    if columnas_categoricas:
        st.markdown("**Distribución de campos categóricos**")
        cols_cat = st.columns(min(2, len(columnas_categoricas)))
        for i, col in enumerate(columnas_categoricas):
            conteo = df[col].fillna("(sin dato)").value_counts().reset_index()
            conteo.columns = [col, "Cantidad"]
            if len(conteo) <= 30:
                with cols_cat[i % len(cols_cat)]:
                    fig = px.bar(
                        conteo, x=col, y="Cantidad", color=col,
                        title=f"Distribución de {col}",
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

    # --- Correlaciones si hay >=2 numéricas ---
    if len(columnas_numericas) >= 2:
        st.markdown("**Correlación entre variables numéricas**")
        corr = df[columnas_numericas].corr().round(2)
        fig_corr = px.imshow(
            corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Matriz de Correlación",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("Pega el texto de las entrevistas y presiona **Extraer datos** para comenzar el EDA.")
