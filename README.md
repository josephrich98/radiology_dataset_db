# Radiology Dataset Database

A pipeline for automatically discovering, extracting, and structuring radiology datasets from the literature.

- `data/`: contains the final dataset table (e.g., `radiology_db.csv`)
- `src/`: scripts for querying PubMed, extracting dataset metadata, and building the database
- `notebooks/`: tutorials and exploratory data analysis notebooks
- `tests/`: pytest-based testing suite

## ⚙️ Installation

### 1. Install and run vLLM (for local LLM inference)

```bash
conda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm
```

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8001 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

### 2. Set up the radiology dataset database environment

```bash
git clone git@github.com:josephrich98/radiology_dataset_db.git
cd radiology_dataset_db
conda create -n radiology_dataset_db python=3.10 -y
conda activate radiology_dataset_db
pip install -e .
```

### 3. Rename `.env_sample` to `.env` and fill in your Entrez email and API key (optional but recommended for higher rate limits)

## 🚀 Usage
python src/build_database_table.py

## Testing
### Just unit tests:
pytest

### Just integration tests:
pytest -m integration

### All tests:
pytest -m ""