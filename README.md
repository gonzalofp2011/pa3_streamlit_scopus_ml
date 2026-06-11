# PA3 - Dashboard Bibliométrico Scopus en Streamlit

Este proyecto implementa un dashboard en Streamlit usando un archivo CSV exportado desde Scopus.

## Pregunta de investigación

¿Cómo se aplican las técnicas de clustering y optimización en Big Data Mining para mejorar la segmentación de clientes, la detección de riesgo y el análisis empresarial?

## Keywords

1. big data clustering
2. K-means
3. data mining
4. customer segmentation

## Query sugerida para Scopus

```text
TITLE-ABS-KEY(("big data clustering" OR "K-means") AND "data mining" AND ("customer segmentation" OR "loan risk" OR business))
```

## Dataset

El repositorio incluye el archivo:

```text
scopus_articles.csv
```

Este archivo contiene 13 artículos exportados desde Scopus con metadatos como autores, título, año, fuente, citas, DOI, abstract y keywords.

## Ejecutar localmente

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Publicar en Streamlit Cloud

1. Subir este proyecto a GitHub.
2. Entrar a Streamlit Cloud.
3. Crear una nueva app.
4. Seleccionar el repositorio.
5. Colocar `app.py` como archivo principal.
6. Deploy.

## Archivos principales

```text
app.py
requirements.txt
scopus_articles.csv
README.md
LICENSE
```

## Licencia

Apache License 2.0.
