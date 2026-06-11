import re
from collections import Counter
from io import StringIO

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Dashboard SCOPUS - Machine Learning",
    page_icon="📚",
    layout="wide"
)

RESEARCH_QUESTION = (
    "¿Cómo se está utilizando Machine Learning para predecir el rendimiento académico "
    "de estudiantes en educación superior entre 2020 y 2026?"
)

KEYWORDS = [
    "machine learning",
    "academic performance",
    "student prediction",
    "higher education"
]

BOOLEAN_QUERY = (
    'TITLE-ABS-KEY("machine learning" AND "academic performance" '
    'AND "student prediction" AND "higher education") '
    'AND PUBYEAR > 2019'
)


@st.cache_data
def load_csv(uploaded_file=None, github_url: str = "") -> pd.DataFrame:
    """
    Loads a Scopus CSV either from a local upload or a public GitHub raw URL.
    """
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    if github_url:
        return pd.read_csv(github_url)

    return pd.read_csv("sample_scopus_ml.csv")


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Finds a column using flexible matching because Scopus exports can vary slightly.
    """
    normalized = {c.lower().strip(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower().strip()
        if key in normalized:
            return normalized[key]

    for col in df.columns:
        col_norm = col.lower().strip()
        for candidate in candidates:
            if candidate.lower().strip() in col_norm:
                return col

    return None


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the most common Scopus columns to a standard structure.
    """
    col_authors = find_column(df, ["Authors", "Author(s)", "Author full names"])
    col_title = find_column(df, ["Title", "Document Title"])
    col_year = find_column(df, ["Year", "Publication Year"])
    col_abstract = find_column(df, ["Abstract", "Abstracts", "Description"])
    col_citations = find_column(df, ["Cited by", "Citations", "Cited by count"])
    col_source = find_column(df, ["Source title", "Journal", "Source"])
    col_doi = find_column(df, ["DOI"])

    normalized = pd.DataFrame()
    normalized["Autores"] = df[col_authors].fillna("No registrado") if col_authors else "No registrado"
    normalized["Título"] = df[col_title].fillna("Sin título") if col_title else "Sin título"
    normalized["Año"] = pd.to_numeric(df[col_year], errors="coerce") if col_year else pd.NA
    normalized["Abstract"] = df[col_abstract].fillna("") if col_abstract else ""
    normalized["Citas"] = pd.to_numeric(df[col_citations], errors="coerce").fillna(0).astype(int) if col_citations else 0
    normalized["Fuente"] = df[col_source].fillna("No registrada") if col_source else "No registrada"
    normalized["DOI"] = df[col_doi].fillna("") if col_doi else ""

    normalized = normalized.dropna(subset=["Año"])
    normalized["Año"] = normalized["Año"].astype(int)

    return normalized


def split_authors(author_text: str) -> list[str]:
    if pd.isna(author_text):
        return []
    parts = re.split(r";|,", str(author_text))
    return [p.strip() for p in parts if p.strip()]


def get_top_authors(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    counter = Counter()
    for authors in df["Autores"]:
        counter.update(split_authors(authors))
    return pd.DataFrame(counter.most_common(top_n), columns=["Autor", "Cantidad"])


def get_word_frequency(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    stopwords = {
        "the", "and", "for", "with", "from", "this", "that", "using", "used",
        "are", "was", "were", "has", "have", "their", "these", "those",
        "study", "research", "results", "paper", "based", "data", "model",
        "models", "students", "student", "learning", "machine", "academic",
        "performance", "education", "higher", "prediction", "predict"
    }

    text = " ".join(df["Abstract"].dropna().astype(str)).lower()
    words = re.findall(r"[a-zA-Z]{4,}", text)
    words = [word for word in words if word not in stopwords]

    counter = Counter(words)
    return pd.DataFrame(counter.most_common(top_n), columns=["Palabra", "Frecuencia"])


st.title("📚 Dashboard de Artículos SCOPUS sobre Machine Learning")
st.caption("Análisis bibliométrico y exploratorio para artículos científicos exportados desde Scopus.")

with st.expander("🔎 Pregunta de investigación y estrategia de búsqueda", expanded=True):
    st.markdown(f"**Pregunta de investigación:** {RESEARCH_QUESTION}")
    st.markdown("**Keywords seleccionadas:**")
    st.write(", ".join(KEYWORDS))
    st.code(BOOLEAN_QUERY, language="text")

st.sidebar.header("Carga del dataset")
uploaded_file = st.sidebar.file_uploader("Sube tu CSV exportado desde Scopus", type=["csv"])
github_url = st.sidebar.text_input(
    "O pega un enlace RAW de GitHub",
    placeholder="https://raw.githubusercontent.com/usuario/repo/main/scopus.csv"
)

try:
    raw_df = load_csv(uploaded_file, github_url)
    df = normalize_dataset(raw_df)
except Exception as exc:
    st.error("No se pudo leer el CSV. Verifica que el archivo tenga formato de exportación Scopus.")
    st.exception(exc)
    st.stop()

if df.empty:
    st.warning("El dataset no contiene registros válidos para analizar.")
    st.stop()

st.sidebar.header("Filtros")
min_year = int(df["Año"].min())
max_year = int(df["Año"].max())
year_range = st.sidebar.slider("Rango de años", min_year, max_year, (min_year, max_year))
keyword_filter = st.sidebar.text_input("Buscar en título o abstract", "")

filtered = df[(df["Año"] >= year_range[0]) & (df["Año"] <= year_range[1])]

if keyword_filter.strip():
    pattern = keyword_filter.strip().lower()
    filtered = filtered[
        filtered["Título"].str.lower().str.contains(pattern, na=False)
        | filtered["Abstract"].str.lower().str.contains(pattern, na=False)
    ]

st.subheader("Indicadores principales")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Artículos", len(filtered))
col2.metric("Años analizados", f"{filtered['Año'].min()} - {filtered['Año'].max()}")
col3.metric("Citas totales", int(filtered["Citas"].sum()))
col4.metric("Promedio de citas", round(filtered["Citas"].mean(), 2))

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Publicaciones por año")
    year_counts = filtered.groupby("Año").size().reset_index(name="Cantidad")
    fig_year = px.bar(year_counts, x="Año", y="Cantidad", text="Cantidad")
    fig_year.update_layout(xaxis_title="Año", yaxis_title="Cantidad de artículos")
    st.plotly_chart(fig_year, use_container_width=True)

with right:
    st.subheader("Top 10 artículos más citados")
    top_cited = filtered.sort_values("Citas", ascending=False).head(10)
    fig_cited = px.bar(
        top_cited,
        x="Citas",
        y="Título",
        orientation="h",
        hover_data=["Autores", "Año", "Fuente"]
    )
    fig_cited.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_cited, use_container_width=True)

left2, right2 = st.columns(2)

with left2:
    st.subheader("Autores con mayor presencia")
    top_authors = get_top_authors(filtered, 10)
    fig_authors = px.bar(top_authors, x="Cantidad", y="Autor", orientation="h")
    fig_authors.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_authors, use_container_width=True)

with right2:
    st.subheader("Palabras frecuentes en abstracts")
    word_freq = get_word_frequency(filtered, 20)
    fig_words = px.bar(word_freq, x="Frecuencia", y="Palabra", orientation="h")
    fig_words.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_words, use_container_width=True)

st.divider()

st.subheader("Interpretación del análisis")
st.markdown(
    """
    El dashboard permite identificar la evolución anual de publicaciones, los trabajos con mayor impacto
    medido por citas, los autores más recurrentes y los conceptos frecuentes en los abstracts.
    Estos elementos ayudan a responder la pregunta de investigación porque muestran tendencias,
    concentración temática y relevancia científica del uso de Machine Learning para predecir
    el rendimiento académico en educación superior.
    """
)

st.subheader("Dataset normalizado")
st.dataframe(filtered, use_container_width=True)

csv_download = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "Descargar dataset filtrado",
    data=csv_download,
    file_name="scopus_filtrado.csv",
    mime="text/csv"
)