# DB2-Proyecto2

Sistema de búsqueda multimodal (audio, imagen, texto) que compara un motor propio en memoria vs PostgreSQL + pgvector.

## Requisitos

- Python 3.10+
- pip

## Instalación

```bash
# Crear entorno virtual (opcional pero recomendado)
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Avance actual — Hito 1 (Mocks)

### ETL Pipeline (escanea datasets)
```bash
python3 scripts/etl_pipeline.py
```

### Benchmark simulado
```bash
python3 scripts/run_experiments.py 100
```

### Demo UI (Streamlit)
```bash
streamlit run ui/app.py
```

## Estructura del proyecto

```
├── scripts/
│   ├── etl_pipeline.py        # Pipeline maestro de extracción
│   └── run_experiments.py     # Benchmarking de rendimiento
├── ui/
│   └── app.py                 # Interfaz visual Streamlit
├── src/
│   ├── api/                   # Endpoints FastAPI
│   ├── db/                    # Conexión y modelos PostgreSQL
│   ├── engine/                # Motor de búsqueda custom
│   ├── extraction/            # Extracción de features
│   └── ml/                    # Codebook y cuantización
├── docker/
│   └── init.sql               # Setup de BD
└── docker-compose.yml         # Infraestructura PostgreSQL
```

## Datasets

Búsqueda Musical Inteligente
https://www.kaggle.com/datasets/imuhammad/audio-features-and-lyrics-of-spotify-songs

Recomendación Multimodal
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

Data de audios
https://github.com/mdeff/fma/tree/master

