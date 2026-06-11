# PA3 - Dashboard SCOPUS y Streamlit

## Pregunta de investigación

¿Cómo se está utilizando Machine Learning para predecir el rendimiento académico de estudiantes en educación superior entre 2020 y 2026?

## Keywords

1. machine learning
2. academic performance
3. student prediction
4. higher education

## Query sugerida para Scopus

```text
TITLE-ABS-KEY("machine learning" AND "academic performance" AND "student prediction" AND "higher education") AND PUBYEAR > 2019
```

## Estructura del proyecto

```text
pa3_streamlit_scopus_ml/
│── app.py
│── sample_scopus_ml.csv
│── requirements.txt
│── README.md
│── LICENSE
```

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Cloud

1. Crear un repositorio público en GitHub.
2. Subir los archivos del proyecto.
3. Ingresar a Streamlit Cloud.
4. Seleccionar el repositorio.
5. Indicar como archivo principal: `app.py`.
6. Deploy.

## Dataset

El archivo `sample_scopus_ml.csv` solo sirve para probar la aplicación. Para la entrega final, reemplazarlo por el CSV exportado desde Scopus con al menos 10 artículos científicos.