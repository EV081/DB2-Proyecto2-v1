# DB2-Proyecto2 — Sistema Multimodal de Recuperación y Búsqueda

Sistema unificado de búsqueda sobre **texto, audio e imagen** que compara dos motores:

- **Motor propio en memoria/disco** — Índice Invertido **SPIMI** + TF-IDF + Similitud Coseno, implementado desde cero.
- **PostgreSQL nativo** — `pgvector` (HNSW) + Full-Text Search (GIN / GiST).

Aplicaciones cubiertas (del enunciado, fase 2):
- **App 2: Búsqueda Musical Inteligente** — letras (texto) + audio (MFCC).
- **App 4: Recomendación Multimodal** — descripciones (texto) + imágenes (SIFT) de productos fashion.

## Stack

- Python 3.10+
- PostgreSQL 16 + pgvector
- Docker Compose
- FastAPI (backend), librosa (audio), OpenCV (imagen), NLTK (texto)

## Setup

### 1. Dependencias Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Variables de entorno

```bash
cp .env.example .env
# editar .env y poner tu POSTGRES_PASSWORD (y actualizar DATABASE_URL al mismo)
```

### 3. Levantar PostgreSQL + pgvector

```bash
docker compose up -d
docker compose ps                       # debería decir "healthy"
```

El schema se crea automáticamente vía `docker/init.sql`. Si necesitas resetear:

```bash
docker compose down -v && docker compose up -d
```

### 4. Credenciales de Kaggle (requerido para Spotify lyrics y Fashion images)

Kaggle exige token incluso para datasets públicos. El único campo que necesitas tocar es `KAGGLE_API_TOKEN` en tu `.env`.

1. Genera el token en <https://www.kaggle.com/settings> → **API** → **Create New Token** (formato nuevo: `KGAT_...`).
2. Pégalo en `.env`:

```bash
# .env
KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

`setup_all.sh` carga `.env` automáticamente y lo exporta a los scripts. Sin token, se saltean Spotify/Fashion y corre solo jam-alt + FMA.

### 5. Setup con un solo comando

```bash
bash scripts/setup_all.sh
```

Hace todo de corrido: descarga (idempotente, vía sentinel `data/<ds>/.downloaded`), levanta Postgres, corre los ETLs y **arranca el backend FastAPI** en <http://127.0.0.1:8000> (docs en `/docs`). Ctrl+C para detener.

| Dataset | Origen | Auth | Tamaño |
|---|---|---|---|
| jam-alt | HuggingFace `jamendolyrics/jam-alt` (filtrado a `en`) | ninguna | ~500 MB |
| FMA small | HTTPS directo `os.unil.cloud.switch.ch/fma/fma_small.zip` | ninguna | ~8 GB |
| Spotify lyrics | Kaggle `imuhammad/audio-features-and-lyrics-of-spotify-songs` | `KAGGLE_API_TOKEN` en `.env` | ~100 MB |
| Fashion images | Kaggle `paramaggarwal/fashion-product-images-dataset` | `KAGGLE_API_TOKEN` en `.env` | ~25 GB |

Flags útiles:

```bash
bash scripts/setup_all.sh --no-serve             # ETL completo sin levantar API
bash scripts/setup_all.sh --port 8080            # cambia puerto
bash scripts/setup_all.sh --force-index          # re-indexa sin re-descargar
bash scripts/setup_all.sh --force-download       # vuelve a bajar todo
bash scripts/setup_all.sh --only jamalt          # solo un dataset
```

## Modos de uso

### A. Demo local con tus archivos reales

Si descargaste el dataset de **Spotify lyrics + FMA**:

```bash
python3 scripts/etl_music.py \
    --lyrics-dir ~/datasets/spotify/lyrics \
    --audio-dir ~/datasets/fma_small \
    --codebook-text 1000 --codebook-audio 200 \
    --index-dir data/music_index \
    --metadata-csv ~/datasets/spotify/metadata.csv \
    --reset
```

Donde `metadata.csv` tiene columnas `stem,title,artist,genre` (todas opcionales menos `stem`).

Si descargaste **Fashion Product Images**:

```bash
python3 scripts/etl_fashion.py \
    --images-dir ~/datasets/fashion/images \
    --descriptions-dir ~/datasets/fashion/descs \
    --codebook-image 128 --codebook-text 1000 \
    --index-dir data/fashion_index \
    --metadata-csv ~/datasets/fashion/styles.csv \
    --reset
```

Donde `styles.csv` tiene columnas `stem,name,category,subcategory`.

**Convención de naming:** los archivos se parean por `stem` (filename sin extensión). Para música, `song_001.txt` (lyrics) y `song_001.mp3` (audio) son la misma canción. Para fashion, `prod_001.jpg` y `prod_001.txt` son el mismo producto.

### B. Levantar la API

```bash
uvicorn src.main:app --reload
```

API en <http://localhost:8000>, docs en <http://localhost:8000/docs>.

#### Endpoints

| Método | Path | Descripción |
|---|---|---|
| `GET`  | `/api/music/search/lyrics?q=...&engine=...&k=...` | Buscar canciones por letra |
| `POST` | `/api/music/search/audio` (multipart `file`) | Buscar canciones por audio similar |
| `GET`  | `/api/fashion/search/description?q=...&engine=...&k=...` | Buscar productos por descripción |
| `POST` | `/api/fashion/search/image` (multipart `file`) | Buscar productos por imagen similar |
| `GET`  | `/health` | Healthcheck |
| `GET`  | `/api/db/status` | Estado de PostgreSQL + pgvector |

**Engines disponibles:**
- `spimi` — motor propio (índice invertido + accumulator)
- `gin` — Postgres full-text GIN
- `gist` — Postgres full-text GiST
- `pgvector` — Postgres HNSW coseno sobre embeddings densos

Para audio/imagen solo `spimi` y `pgvector` (GIN/GiST son específicos de full-text).

#### Ejemplos

```bash
# Texto
curl 'http://localhost:8000/api/music/search/lyrics?q=love+heart&engine=spimi&k=5'
curl 'http://localhost:8000/api/fashion/search/description?q=blue+shirt&engine=gin&k=5'

# Audio
curl -F 'file=@query_song.mp3' \
     'http://localhost:8000/api/music/search/audio?engine=pgvector&k=5'

# Imagen
curl -F 'file=@query_product.jpg' \
     'http://localhost:8000/api/fashion/search/image?engine=spimi&k=5'
```

## Benchmark (Fase 4)

Después de poblar la BD con un ETL real:

```bash
# Comparativa SPIMI vs pgvector vs GIN/GiST con latencia + throughput
python3 scripts/bench_full.py --queries 100 --k 10

# Recall@K con ground truth de metadata
python3 scripts/compute_recall.py --queries 50 --k 10
```

Outputs:
- `benchmark_fase4.json` + `benchmark_fase4.md`
- `recall_fase4.json`

### Benchmark del SPIMI puro (sin BD)

Para medir el motor propio aislado en cargas 1K/10K/100K:

```bash
python3 scripts/gen_synthetic_corpus.py --n 1000   --out data/synthetic_1k.jsonl
python3 scripts/gen_synthetic_corpus.py --n 10000  --out data/synthetic_10k.jsonl
python3 scripts/gen_synthetic_corpus.py --n 100000 --out data/synthetic_100k.jsonl
python3 scripts/bench_spimi.py --queries 100
```

## Tests

```bash
PYTHONPATH=. python3 tests/tests_inverted_index.py        # SPIMI puro
PYTHONPATH=. python3 tests/tests_integration_text.py      # pipeline texto
PYTHONPATH=. python3 tests/tests_integration_image.py     # pipeline imagen
PYTHONPATH=. python3 tests/tests_integration_audio.py     # pipeline audio
PYTHONPATH=. python3 tests/tests_similarity.py            # math TF-IDF/coseno
PYTHONPATH=. python3 tests/tests_extraction.py            # extractores
```

## Estructura

```
src/
├── extraction/         # SIFT (imagen), MFCC (audio), TF-IDF+NLTK (texto)
├── ml/                 # KClustering (K-Means/Medoids), TokKWords, VectorQuantizer
├── engine/             # SPIMI (inverted_index.py), pipelines por modalidad
│   ├── inverted_index.py    # PostingList + spimi_invert + merge + InvertedIndex
│   ├── similarity.py        # TF-IDF + cosine + minkowski + top-K denso
│   ├── text_pipeline.py     # extracción → codebook → SPIMI (texto)
│   ├── audio_pipeline.py    # MFCC → K-Means → SPIMI (audio)
│   └── image_pipeline.py    # SIFT → K-Means → SPIMI (imagen)
├── db/                 # PostgreSQL: schema, native_search, storage
└── api/                # FastAPI: routes + search_service dispatcher

scripts/
├── etl_music.py            # ETL App 2
├── etl_fashion.py          # ETL App 4
├── bench_spimi.py          # benchmark SPIMI puro (1K/10K/100K sintéticos)
├── bench_full.py           # benchmark 4 motores sobre BD real
├── compute_recall.py       # recall@K con metadata como ground truth
└── gen_synthetic_corpus.py # generador de corpus sintético JSONL
```

## Datasets sugeridos

- **App 2 — música:**
  - [Audio features and lyrics of Spotify songs](https://www.kaggle.com/datasets/imuhammad/audio-features-and-lyrics-of-spotify-songs)
  - [FMA: A Dataset For Music Analysis](https://github.com/mdeff/fma)
- **App 4 — fashion:**
  - [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)

## Limitaciones conocidas

- El K-Means en `src/ml/clustering_trainer.py` está en Python puro sin paralelización; para corpus muy grandes (decenas de miles de vectores SIFT/MFCC) puede tomar minutos.
- La cuantización de audio se hace sobre el archivo completo, sin sliding window de codewords múltiples.
- Recall@K requiere que se cargue metadata (artist/genre o category/subcategory) vía `--metadata-csv`; sin ella el score es `null`.
