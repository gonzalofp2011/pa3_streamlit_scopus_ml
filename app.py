import re
from collections import Counter
from io import StringIO

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Dashboard Scopus - Machine Learning",
    page_icon="📚",
    layout="wide"
)

RESEARCH_QUESTION = (
    "¿Cómo se aplica el Machine Learning para mejorar la predicción de enfermedades "
    "cardiovasculares usando datos clínicos en artículos científicos indexados en Scopus?"
)

KEYWORDS = ["machine learning", "cardiovascular disease", "prediction", "clinical data"]
BOOLEAN_QUERY = 'TITLE-ABS-KEY("machine learning" AND "cardiovascular disease" AND prediction AND "clinical data")'

@st.cache_data
def load_csv(uploaded_file=None, github_url: str = "") -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if github_url.strip():
        return pd.read_csv(github_url.strip())
    return pd.read_csv("scopus_articles.csv")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        clean = col.strip().lower()
        if clean in ["authors", "author(s)", "autores"]:
            mapping[col] = "Authors"
        elif clean in ["title", "titulo", "título", "document title"]:
            mapping[col] = "Title"
        elif clean in ["year", "año", "publication year"]:
            mapping[col] = "Year"
        elif clean in ["abstract", "resumen"]:
            mapping[col] = "Abstract"
        elif clean in ["cited by", "citations", "citas", "cited_by"]:
            mapping[col] = "Cited by"
        elif clean in ["source title", "journal", "revista"]:
            mapping[col] = "Source title"
        elif clean == "doi":
            mapping[col] = "DOI"
    df = df.rename(columns=mapping)
    return df


def get_required_columns(df: pd.DataFrame):
    required = ["Authors", "Title", "Year", "Abstract", "Cited by"]
    return [col for col in required if col not in df.columns]


def split_authors(authors_text: str):
    if pd.isna(authors_text):
        return []
    parts = re.split(r";|, and | and ", str(authors_text))
    return [p.strip() for p in parts if p.strip()]


def clean_words(text_series: pd.Series):
    stopwords = {
        "the", "and", "for", "with", "from", "this", "that", "are", "was", "were", "using",
        "used", "into", "their", "have", "has", "been", "data", "study", "model", "models",
        "learning", "machine", "big", "data", "clustering", "cluster", "clusters", "kmeans",
        "means", "mining", "business", "method", "algorithm", "algorithms", "analysis",
        "using", "based", "model", "models", "approach",
        "una", "unos", "las", "los", "para", "con", "del", "por", "que", "como"
    }
    text = " ".join(text_series.fillna("").astype(str)).lower()
    words = re.findall(r"[a-záéíóúñ]{4,}", text)
    words = [w for w in words if w not in stopwords]
    return Counter(words).most_common(20)


st.title("📚 Dashboard Bibliométrico SCOPUS - Big Data Clustering")
st.markdown(
    "Este dashboard analiza artículos científicos exportados desde Scopus y permite explorar "
    "publicaciones, citas, autores y palabras frecuentes en los abstracts."
)

with st.expander("🎯 Pregunta de investigación y keywords", expanded=True):
    st.subheader("Pregunta de investigación")
    st.write(RESEARCH_QUESTION)
    st.subheader("4 Keywords")
    st.write(", ".join([f"**{k}**" for k in KEYWORDS]))
    st.subheader("Cadena de búsqueda sugerida para Scopus")
    st.code(BOOLEAN_QUERY, language="text")

st.sidebar.header("Carga de datos")
uploaded_file = st.sidebar.file_uploader("Sube tu CSV exportado de Scopus", type=["csv"])
github_url = st.sidebar.text_input("O pega la URL RAW del CSV en GitHub")

try:
    df = normalize_columns(load_csv(uploaded_file, github_url))
except Exception as e:
    st.error(f"No se pudo cargar el archivo CSV: {e}")
    st.stop()

missing = get_required_columns(df)
if missing:
    st.error("Faltan columnas necesarias: " + ", ".join(missing))
    st.info("Columnas mínimas esperadas: Authors, Title, Year, Abstract, Cited by.")
    st.stop()

# Limpieza básica
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Cited by"] = pd.to_numeric(df["Cited by"], errors="coerce").fillna(0).astype(int)
df = df.dropna(subset=["Title", "Year"])
df["Year"] = df["Year"].astype(int)

st.sidebar.header("Filtros")
year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
year_range = st.sidebar.slider("Rango de años", year_min, year_max, (year_min, year_max))
min_cites = st.sidebar.number_input("Citas mínimas", min_value=0, value=0, step=1)
keyword_filter = st.sidebar.text_input("Buscar palabra en título o abstract")

filtered = df[(df["Year"].between(year_range[0], year_range[1])) & (df["Cited by"] >= min_cites)]
if keyword_filter.strip():
    pattern = keyword_filter.strip().lower()
    filtered = filtered[
        filtered["Title"].fillna("").str.lower().str.contains(pattern, regex=False)
        | filtered["Abstract"].fillna("").str.lower().str.contains(pattern, regex=False)
    ]

st.subheader("Indicadores principales")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Artículos", len(filtered))
col2.metric("Años analizados", f"{filtered['Year'].min()} - {filtered['Year'].max()}" if len(filtered) else "-")
col3.metric("Total de citas", int(filtered["Cited by"].sum()) if len(filtered) else 0)
col4.metric("Promedio de citas", round(filtered["Cited by"].mean(), 2) if len(filtered) else 0)

if filtered.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()

st.subheader("Visualizaciones")

pub_year = filtered.groupby("Year", as_index=False).size().rename(columns={"size": "Publicaciones"})
fig_year = px.bar(pub_year, x="Year", y="Publicaciones", title="Distribución de publicaciones por año")
st.plotly_chart(fig_year, use_container_width=True)

top_cited = filtered.sort_values("Cited by", ascending=False).head(10)
fig_cites = px.bar(
    top_cited,
    x="Cited by",
    y="Title",
    orientation="h",
    title="Top 10 artículos más citados",
    hover_data=["Authors", "Year"]
)
fig_cites.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig_cites, use_container_width=True)

all_authors = []
for authors in filtered["Authors"]:
    all_authors.extend(split_authors(authors))
authors_df = pd.DataFrame(Counter(all_authors).most_common(10), columns=["Autor", "Frecuencia"])
fig_authors = px.bar(authors_df, x="Frecuencia", y="Autor", orientation="h", title="Autores con mayor frecuencia")
fig_authors.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig_authors, use_container_width=True)

words_df = pd.DataFrame(clean_words(filtered["Abstract"]), columns=["Palabra", "Frecuencia"])
fig_words = px.bar(words_df, x="Palabra", y="Frecuencia", title="Palabras frecuentes en abstracts")
st.plotly_chart(fig_words, use_container_width=True)


if "Document Type" in filtered.columns:
    doc_types = filtered["Document Type"].fillna("Sin tipo").value_counts().reset_index()
    doc_types.columns = ["Tipo de documento", "Cantidad"]
    fig_doc = px.bar(doc_types, x="Tipo de documento", y="Cantidad", title="Tipo de documento")
    st.plotly_chart(fig_doc, use_container_width=True)

if "Author Keywords" in filtered.columns:
    keyword_text = filtered["Author Keywords"].fillna("").astype(str).str.replace(";", ",")
    keyword_counter = Counter()
    for row in keyword_text:
        for kw in row.split(","):
            kw = kw.strip()
            if kw:
                keyword_counter[kw] += 1
    kw_df = pd.DataFrame(keyword_counter.most_common(15), columns=["Keyword", "Frecuencia"])
    if not kw_df.empty:
        fig_kw = px.bar(kw_df, x="Frecuencia", y="Keyword", orientation="h", title="Keywords de autores más frecuentes")
        fig_kw.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_kw, use_container_width=True)

if "Source title" in filtered.columns:
    sources = filtered["Source title"].fillna("Sin fuente").value_counts().head(10).reset_index()
    sources.columns = ["Fuente", "Artículos"]
    fig_sources = px.pie(sources, names="Fuente", values="Artículos", title="Principales revistas o fuentes")
    st.plotly_chart(fig_sources, use_container_width=True)

st.subheader("Tabla de artículos filtrados")
show_cols = [c for c in ["Authors", "Title", "Year", "Source title", "Cited by", "DOI", "Abstract"] if c in filtered.columns]
st.dataframe(filtered[show_cols], use_container_width=True)

st.subheader("Conclusión automática")
top_year = pub_year.sort_values("Publicaciones", ascending=False).iloc[0]["Year"]
top_article = top_cited.iloc[0]["Title"]
st.success(
    f"El análisis muestra {len(filtered)} artículos relacionados con la pregunta de investigación. "
    f"El año con mayor producción fue {top_year}. El artículo con mayor impacto por citas es: '{top_article}'. "
    "Las palabras frecuentes en los abstracts permiten identificar los enfoques técnicos predominantes del campo."
)
