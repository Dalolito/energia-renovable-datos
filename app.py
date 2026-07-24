import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------
# Configuración general de la página
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Energía Renovable",
    page_icon="⚡",
    layout="wide",
)

# ------------------------------------------------------------------
# Carga de datos
# ------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv("energia_renovable.csv")
    df["Fecha_Entrada_Operacion"] = pd.to_datetime(
        df["Fecha_Entrada_Operacion"], errors="coerce"
    )
    df["MUSD_por_MWh_dia"] = df["Inversion_Inicial_MUSD"] / df["Generacion_Diaria_MWh"]
    df["Año_Entrada"] = df["Fecha_Entrada_Operacion"].dt.year
    return df

df = cargar_datos()

# ------------------------------------------------------------------
# Sidebar - Filtros
# ------------------------------------------------------------------
st.sidebar.header("Filtros")

tecnologias = sorted(df["Tecnologia"].unique())
tecnologias_sel = st.sidebar.multiselect(
    "Tecnología", tecnologias, default=tecnologias
)

operadores = sorted(df["Operador"].unique())
operadores_sel = st.sidebar.multiselect(
    "Operador", operadores, default=operadores
)

estados = sorted(df["Estado_Actual"].unique())
estados_sel = st.sidebar.multiselect(
    "Estado Actual", estados, default=estados
)

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
# Encabezado
# ------------------------------------------------------------------
st.title("⚡ Dashboard de Energía Renovable")
st.markdown("Panel interactivo de proyectos **Solares, Eólicos y PCH** conectados al SIN.")

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
# Pregunta de negocio: ¿Qué tecnología tiene la mejor relación
# Inversión vs. Generación Diaria?
# ------------------------------------------------------------------
st.subheader("💰 Inversión vs. Generación Diaria por Tecnología")

col_izq, col_der = st.columns([2, 1])

with col_izq:
    fig_scatter = px.scatter(
        df_filtrado,
        x="Inversion_Inicial_MUSD",
        y="Generacion_Diaria_MWh",
        color="Tecnologia",
        size="Capacidad_Instalada_MW",
        hover_data=["ID_Proyecto", "Operador", "Estado_Actual"],
        title="Inversión vs. Generación Diaria por Proyecto",
        labels={
            "Inversion_Inicial_MUSD": "Inversión Inicial (MUSD)",
            "Generacion_Diaria_MWh": "Generación Diaria (MWh)",
        },
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_der:
    resumen_eficiencia = (
        df_filtrado.groupby("Tecnologia")
        .agg(
            Inversion_Total_MUSD=("Inversion_Inicial_MUSD", "sum"),
            Generacion_Total_MWh=("Generacion_Diaria_MWh", "sum"),
        )
        .reset_index()
    )
    resumen_eficiencia["MUSD_por_MWh_dia"] = (
        resumen_eficiencia["Inversion_Total_MUSD"] / resumen_eficiencia["Generacion_Total_MWh"]
    )
    resumen_eficiencia = resumen_eficiencia.sort_values("MUSD_por_MWh_dia")

    fig_barras_ef = px.bar(
        resumen_eficiencia,
        x="Tecnologia",
        y="MUSD_por_MWh_dia",
        color="Tecnologia",
        text_auto=".2f",
        title="MUSD por MWh diario (menor = mejor)",
        labels={"MUSD_por_MWh_dia": "MUSD / MWh día"},
    )
    fig_barras_ef.update_layout(showlegend=False)
    st.plotly_chart(fig_barras_ef, use_container_width=True)

    mejor_tec = resumen_eficiencia.iloc[0]["Tecnologia"]
    st.success(f"✅ **{mejor_tec}** tiene la mejor relación Inversión / Generación Diaria.")

st.divider()

# ------------------------------------------------------------------
# Visualización clave: Capacidad Instalada por Operador
# ------------------------------------------------------------------
st.subheader("🏗️ Capacidad Instalada por Operador")

capacidad_operador = (
    df_filtrado.groupby("Operador")["Capacidad_Instalada_MW"]
    .sum()
    .reset_index()
    .sort_values("Capacidad_Instalada_MW", ascending=False)
)

fig_operador = px.bar(
    capacidad_operador,
    x="Operador",
    y="Capacidad_Instalada_MW",
    color="Operador",
    text_auto=".0f",
    title="Capacidad Instalada Total por Operador (MW)",
    labels={"Capacidad_Instalada_MW": "Capacidad Instalada (MW)"},
)
fig_operador.update_layout(showlegend=False)
st.plotly_chart(fig_operador, use_container_width=True)

st.divider()

# ------------------------------------------------------------------
# Información general adicional
# ------------------------------------------------------------------
st.subheader("📊 Contexto General")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Distribución por Tecnología", "Estado de Proyectos", "Evolución Temporal", "Eficiencia de Planta"]
)

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        fig_pie_tec = px.pie(
            df_filtrado,
            names="Tecnologia",
            values="Capacidad_Instalada_MW",
            title="Participación de Capacidad Instalada por Tecnología",
            hole=0.4,
        )
        st.plotly_chart(fig_pie_tec, use_container_width=True)
    with col_b:
        fig_count_tec = px.bar(
            df_filtrado["Tecnologia"].value_counts().reset_index(),
            x="Tecnologia",
            y="count",
            color="Tecnologia",
            title="Número de Proyectos por Tecnología",
            labels={"count": "Cantidad de Proyectos"},
        )
        fig_count_tec.update_layout(showlegend=False)
        st.plotly_chart(fig_count_tec, use_container_width=True)

with tab2:
    col_c, col_d = st.columns(2)
    with col_c:
        fig_estado = px.pie(
            df_filtrado,
            names="Estado_Actual",
            title="Distribución de Proyectos por Estado Actual",
            hole=0.4,
        )
        st.plotly_chart(fig_estado, use_container_width=True)
    with col_d:
        fig_estado_tec = px.bar(
            df_filtrado.groupby(["Estado_Actual", "Tecnologia"]).size().reset_index(name="Cantidad"),
            x="Estado_Actual",
            y="Cantidad",
            color="Tecnologia",
            barmode="group",
            title="Estado Actual por Tecnología",
        )
        st.plotly_chart(fig_estado_tec, use_container_width=True)

with tab3:
    evolucion = (
        df_filtrado.dropna(subset=["Año_Entrada"])
        .groupby(["Año_Entrada", "Tecnologia"])["Capacidad_Instalada_MW"]
        .sum()
        .reset_index()
    )
    fig_evolucion = px.bar(
        evolucion,
        x="Año_Entrada",
        y="Capacidad_Instalada_MW",
        color="Tecnologia",
        title="Capacidad Instalada Nueva por Año de Entrada en Operación",
        labels={"Capacidad_Instalada_MW": "Capacidad Instalada (MW)", "Año_Entrada": "Año"},
    )
    st.plotly_chart(fig_evolucion, use_container_width=True)

with tab4:
    fig_box_ef = px.box(
        df_filtrado,
        x="Tecnologia",
        y="Eficiencia_Planta_Pct",
        color="Tecnologia",
        title="Distribución de Eficiencia de Planta (%) por Tecnología",
        points="all",
    )
    fig_box_ef.update_layout(showlegend=False)
    st.plotly_chart(fig_box_ef, use_container_width=True)

st.divider()

# ------------------------------------------------------------------
# Tabla de datos filtrados
# ------------------------------------------------------------------
st.subheader("📋 Datos Detallados")
st.dataframe(df_filtrado, use_container_width=True)

st.caption(f"Mostrando {len(df_filtrado):,} de {len(df):,} proyectos totales.")
