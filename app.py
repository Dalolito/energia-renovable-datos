import os
import json

import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq

# ------------------------------------------------------------------
# Configuración general de la página
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Inteligente - Energía Renovable",
    page_icon="⚡",
    layout="wide",
)

CSV_PATH = "energia_renovable.csv"
MODELO = "llama-3.3-70b-versatile"

# ------------------------------------------------------------------
# Carga de datos
# ------------------------------------------------------------------
@st.cache_data
def procesar_datos(df):
    df["Fecha_Entrada_Operacion"] = pd.to_datetime(
        df["Fecha_Entrada_Operacion"], errors="coerce"
    )
    df["MUSD_por_MWh_dia"] = df["Inversion_Inicial_MUSD"] / df["Generacion_Diaria_MWh"]
    df["Año_Entrada"] = df["Fecha_Entrada_Operacion"].dt.year
    return df

if os.path.exists(CSV_PATH):
    df = procesar_datos(pd.read_csv(CSV_PATH))
else:
    st.warning(f"No se encontró **{CSV_PATH}**. Súbelo para continuar.")
    archivo_subido = st.file_uploader("Sube el archivo CSV", type=["csv"])
    if archivo_subido is None:
        st.stop()
    df = procesar_datos(pd.read_csv(archivo_subido))

# ------------------------------------------------------------------
# Sidebar - Filtros + API Key
# ------------------------------------------------------------------
st.sidebar.header("🔑 Groq API")


def obtener_api_key():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = None
    return api_key


api_key_env = obtener_api_key()
api_key = st.sidebar.text_input(
    "Groq API Key",
    value=api_key_env if api_key_env else "",
    type="password",
    placeholder="gsk_...",
)
if api_key:
    st.sidebar.success("API Key cargada ✅")
else:
    st.sidebar.warning("Sin API Key, el chat no funcionará.")

st.sidebar.divider()
st.sidebar.header("Filtros")

tecnologias = sorted(df["Tecnologia"].unique())
tecnologias_sel = st.sidebar.multiselect("Tecnología", tecnologias, default=tecnologias)

operadores = sorted(df["Operador"].unique())
operadores_sel = st.sidebar.multiselect("Operador", operadores, default=operadores)

estados = sorted(df["Estado_Actual"].unique())
estados_sel = st.sidebar.multiselect("Estado Actual", estados, default=estados)

conectado_sel = st.sidebar.multiselect(
    "Conectado al SIN",
    options=df["Conectado_SIN"].unique().tolist(),
    default=df["Conectado_SIN"].unique().tolist(),
)

df_filtrado = df[
    df["Tecnologia"].isin(tecnologias_sel)
    & df["Operador"].isin(operadores_sel)
    & df["Estado_Actual"].isin(estados_sel)
    & df["Conectado_SIN"].isin(conectado_sel)
]

if df_filtrado.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()

# ------------------------------------------------------------------
# Construcción del "resumen de contexto" para el LLM
# Esto es lo que hace que el modelo pueda "ver" e interpretar el
# dashboard: le mandamos estadísticas agregadas (no los 500 registros
# crudos) como contexto en cada pregunta.
# ------------------------------------------------------------------
def construir_contexto(data: pd.DataFrame) -> str:
    resumen_tec = (
        data.groupby("Tecnologia")
        .agg(
            proyectos=("ID_Proyecto", "count"),
            capacidad_total_mw=("Capacidad_Instalada_MW", "sum"),
            generacion_total_mwh=("Generacion_Diaria_MWh", "sum"),
            generacion_promedio_mwh=("Generacion_Diaria_MWh", "mean"),
            eficiencia_promedio_pct=("Eficiencia_Planta_Pct", "mean"),
            inversion_total_musd=("Inversion_Inicial_MUSD", "sum"),
        )
        .round(2)
        .reset_index()
    )
    resumen_tec["musd_por_mwh_dia"] = (
        resumen_tec["inversion_total_musd"] / resumen_tec["generacion_total_mwh"]
    ).round(3)

    resumen_operador = (
        data.groupby("Operador")
        .agg(
            proyectos=("ID_Proyecto", "count"),
            capacidad_total_mw=("Capacidad_Instalada_MW", "sum"),
        )
        .round(2)
        .reset_index()
        .sort_values("capacidad_total_mw", ascending=False)
    )

    resumen_estado = data["Estado_Actual"].value_counts().to_dict()

    resumen_general = {
        "total_proyectos": int(data["ID_Proyecto"].nunique()),
        "capacidad_instalada_total_mw": round(float(data["Capacidad_Instalada_MW"].sum()), 2),
        "generacion_diaria_total_mwh": round(float(data["Generacion_Diaria_MWh"].sum()), 2),
        "inversion_total_musd": round(float(data["Inversion_Inicial_MUSD"].sum()), 2),
        "eficiencia_promedio_pct": round(float(data["Eficiencia_Planta_Pct"].mean()), 2),
        "proyectos_conectados_sin": int(data["Conectado_SIN"].sum()),
        "rango_fechas_entrada": [
            str(data["Fecha_Entrada_Operacion"].min().date())
            if data["Fecha_Entrada_Operacion"].notna().any()
            else None,
            str(data["Fecha_Entrada_Operacion"].max().date())
            if data["Fecha_Entrada_Operacion"].notna().any()
            else None,
        ],
    }

    contexto = {
        "resumen_general": resumen_general,
        "por_tecnologia": resumen_tec.to_dict(orient="records"),
        "por_operador": resumen_operador.to_dict(orient="records"),
        "por_estado_actual": resumen_estado,
    }
    return json.dumps(contexto, ensure_ascii=False, indent=2)


contexto_json = construir_contexto(df_filtrado)

SYSTEM_PROMPT = f"""Eres un analista de datos experto en energía renovable.
Tienes acceso a un resumen estadístico del dashboard actual (ya filtrado según
lo que el usuario seleccionó en pantalla). Usa ÚNICAMENTE esta información
para responder preguntas sobre los datos. Si algo no se puede calcular con
los datos disponibles, dilo claramente en vez de inventar cifras.

Responde de forma clara, breve y con números concretos cuando aplique.
Si te preguntan por "la mejor" o "la peor" tecnología/operador, explica el
criterio que usaste (ej. menor MUSD por MWh diario, mayor capacidad, etc).

CONTEXTO DE DATOS ACTUAL (JSON):
{contexto_json}
"""

# ------------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------------
st.title("⚡ Dashboard Inteligente de Energía Renovable")
st.markdown("Explora los datos y **pregúntale directamente al dashboard** en lenguaje natural.")

# ------------------------------------------------------------------
# KPIs principales
# ------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Generación (MWh/día)", f"{df_filtrado['Generacion_Diaria_MWh'].sum():,.0f}")
col2.metric("Capacidad Instalada Total (MW)", f"{df_filtrado['Capacidad_Instalada_MW'].sum():,.0f}")
col3.metric("Inversión Total (MUSD)", f"{df_filtrado['Inversion_Inicial_MUSD'].sum():,.1f}")
col4.metric("Proyectos", f"{df_filtrado['ID_Proyecto'].nunique():,}")

st.divider()

# ------------------------------------------------------------------
# Tabs: Visualizaciones | Chat con IA
# ------------------------------------------------------------------
tab_viz, tab_chat = st.tabs(["📊 Visualizaciones", "🤖 Pregúntale al Dashboard"])

# ==================== TAB VISUALIZACIONES ====================
with tab_viz:
    st.subheader("💰 Inversión vs. Generación Diaria por Tecnología")

    resumen_eficiencia = (
        df_filtrado.groupby("Tecnologia")
        .agg(
            Inversion_Total_MUSD=("Inversion_Inicial_MUSD", "sum"),
            Generacion_Total_MWh=("Generacion_Diaria_MWh", "sum"),
            Inversion_Promedio_MUSD=("Inversion_Inicial_MUSD", "mean"),
            Generacion_Promedio_MWh=("Generacion_Diaria_MWh", "mean"),
            Proyectos=("ID_Proyecto", "count"),
        )
        .reset_index()
    )
    resumen_eficiencia["MUSD_por_MWh_dia"] = (
        resumen_eficiencia["Inversion_Total_MUSD"] / resumen_eficiencia["Generacion_Total_MWh"]
    )
    resumen_eficiencia = resumen_eficiencia.sort_values("MUSD_por_MWh_dia", ascending=True)
    mejor_tec = resumen_eficiencia.iloc[0]["Tecnologia"]

    col_izq, col_der = st.columns(2)

    with col_izq:
        resumen_eficiencia["Destacado"] = resumen_eficiencia["Tecnologia"].apply(
            lambda t: "Mejor opción" if t == mejor_tec else "Otras"
        )
        fig_ranking = px.bar(
            resumen_eficiencia.sort_values("MUSD_por_MWh_dia", ascending=True),
            x="MUSD_por_MWh_dia",
            y="Tecnologia",
            orientation="h",
            color="Destacado",
            color_discrete_map={"Mejor opción": "#2ecc71", "Otras": "#bdc3c7"},
            text_auto=".3f",
            title="Ranking: Inversión requerida por cada MWh diario (menor = mejor)",
            labels={"MUSD_por_MWh_dia": "MUSD por MWh diario", "Tecnologia": ""},
        )
        fig_ranking.update_layout(showlegend=False, yaxis={"categoryorder": "total descending"})
        st.plotly_chart(fig_ranking, use_container_width=True)

    with col_der:
        fig_frontera = px.scatter(
            resumen_eficiencia,
            x="Inversion_Promedio_MUSD",
            y="Generacion_Promedio_MWh",
            color="Tecnologia",
            text="Tecnologia",
            size="Proyectos",
            size_max=40,
            title="Promedio por Tecnología: menos Inversión y más Generación = mejor",
            labels={
                "Inversion_Promedio_MUSD": "Inversión Promedio (MUSD)",
                "Generacion_Promedio_MWh": "Generación Promedio Diaria (MWh)",
            },
        )
        fig_frontera.update_traces(textposition="top center")
        fig_frontera.add_annotation(
            text="⭐ Zona ideal: arriba-izquierda",
            xref="paper", yref="paper", x=0.02, y=0.98,
            showarrow=False, font=dict(size=11, color="gray"),
        )
        fig_frontera.update_layout(showlegend=False)
        st.plotly_chart(fig_frontera, use_container_width=True)

    st.success(
        f"✅ **{mejor_tec}** tiene la mejor relación Inversión / Generación Diaria: requiere "
        f"solo **{resumen_eficiencia.iloc[0]['MUSD_por_MWh_dia']:.3f} MUSD** por cada MWh "
        f"generado al día."
    )

    st.divider()

    st.subheader("🏗️ Capacidad Instalada por Operador")
    capacidad_operador = (
        df_filtrado.groupby("Operador")["Capacidad_Instalada_MW"]
        .sum()
        .reset_index()
        .sort_values("Capacidad_Instalada_MW", ascending=True)
    )
    fig_operador = px.bar(
        capacidad_operador,
        x="Capacidad_Instalada_MW",
        y="Operador",
        orientation="h",
        color="Capacidad_Instalada_MW",
        color_continuous_scale="Greens",
        text_auto=".0f",
        title="Capacidad Instalada Total por Operador (MW)",
        labels={"Capacidad_Instalada_MW": "Capacidad Instalada (MW)", "Operador": ""},
    )
    fig_operador.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_operador, use_container_width=True)

    st.divider()

    st.subheader("📊 Contexto General")
    sub1, sub2, sub3, sub4 = st.tabs(
        ["Distribución por Tecnología", "Estado de Proyectos", "Evolución Temporal", "Eficiencia de Planta"]
    )
    with sub1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.pie(df_filtrado, names="Tecnologia", values="Capacidad_Instalada_MW",
                       title="Participación de Capacidad por Tecnología", hole=0.4),
                use_container_width=True,
            )
        with c2:
            conteo = df_filtrado["Tecnologia"].value_counts().reset_index()
            st.plotly_chart(
                px.bar(conteo, x="Tecnologia", y="count", color="Tecnologia",
                       title="Número de Proyectos por Tecnología",
                       labels={"count": "Cantidad"}).update_layout(showlegend=False),
                use_container_width=True,
            )
    with sub2:
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(
                px.pie(df_filtrado, names="Estado_Actual", title="Distribución por Estado", hole=0.4),
                use_container_width=True,
            )
        with c4:
            estado_tec = df_filtrado.groupby(["Estado_Actual", "Tecnologia"]).size().reset_index(name="Cantidad")
            st.plotly_chart(
                px.bar(estado_tec, x="Estado_Actual", y="Cantidad", color="Tecnologia",
                       barmode="group", title="Estado Actual por Tecnología"),
                use_container_width=True,
            )
    with sub3:
        evolucion = (
            df_filtrado.dropna(subset=["Año_Entrada"])
            .groupby(["Año_Entrada", "Tecnologia"])["Capacidad_Instalada_MW"]
            .sum()
            .reset_index()
        )
        st.plotly_chart(
            px.bar(evolucion, x="Año_Entrada", y="Capacidad_Instalada_MW", color="Tecnologia",
                   title="Capacidad Instalada Nueva por Año",
                   labels={"Capacidad_Instalada_MW": "Capacidad (MW)", "Año_Entrada": "Año"}),
            use_container_width=True,
        )
    with sub4:
        st.plotly_chart(
            px.box(df_filtrado, x="Tecnologia", y="Eficiencia_Planta_Pct", color="Tecnologia",
                   title="Distribución de Eficiencia de Planta (%)", points="all").update_layout(showlegend=False),
            use_container_width=True,
        )

    st.divider()
    st.subheader("📋 Datos Detallados")
    st.dataframe(df_filtrado, use_container_width=True)
    st.caption(f"Mostrando {len(df_filtrado):,} de {len(df):,} proyectos totales.")

# ==================== TAB CHAT ====================
with tab_chat:
    st.subheader("🤖 Pregúntale al Dashboard")
    st.caption(
        "El modelo responde usando un resumen estadístico de los datos filtrados "
        "actualmente en el panel (no inventa cifras fuera de ese resumen)."
    )

    if not api_key:
        st.warning("Ingresa tu Groq API Key en la barra lateral para poder chatear.")
        st.stop()

    client = Groq(api_key=api_key)

    if "mensajes_dash" not in st.session_state:
        st.session_state.mensajes_dash = []

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("🗑️ Limpiar chat"):
            st.session_state.mensajes_dash = []
            st.rerun()

    for m in st.session_state.mensajes_dash:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    st.markdown("**Preguntas sugeridas:**")
    sugeridas = [
        "¿Qué tecnología tiene la mejor relación inversión vs generación?",
        "¿Cuál operador tiene más capacidad instalada?",
        "¿Cuántos proyectos están en mantenimiento?",
        "Dame un resumen ejecutivo de los datos filtrados",
    ]
    cols = st.columns(2)
    pregunta_sugerida = None
    for i, s in enumerate(sugeridas):
        if cols[i % 2].button(s, key=f"sug_{i}"):
            pregunta_sugerida = s

    pregunta = st.chat_input("Escribe tu pregunta sobre los datos...") or pregunta_sugerida

    if pregunta:
        st.session_state.mensajes_dash.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        mensajes_api = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.mensajes_dash

        with st.chat_message("assistant"):
            placeholder = st.empty()
            respuesta_completa = ""
            try:
                stream = client.chat.completions.create(
                    model=MODELO,
                    messages=mensajes_api,
                    temperature=0.3,
                    max_tokens=1024,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    respuesta_completa += delta
                    placeholder.markdown(respuesta_completa + "▌")
                placeholder.markdown(respuesta_completa)
            except Exception as e:
                respuesta_completa = f"⚠️ Error al llamar a la API de Groq: {e}"
                placeholder.markdown(respuesta_completa)

        st.session_state.mensajes_dash.append({"role": "assistant", "content": respuesta_completa})
