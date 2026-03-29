# Radiology Dataset Database

A pipeline for automatically discovering, extracting, and structuring radiology datasets from the literature.

- `data/`: contains the final dataset table (e.g., `radiology_db.csv`)
- `notebooks/`: tutorials and exploratory data analysis notebooks
- `scripts/`: scripts for running the database building pipeline
- `radiology_dataset_db/`: source code for querying PubMed, extracting dataset metadata, and building the database
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
git clone git@github.com:pachterlab/radiology_dataset_db.git
cd radiology_dataset_db
conda create -n radiology_dataset_db python=3.10 -y
conda activate radiology_dataset_db
pip install -e .
```

### 3. Rename `.env_sample` to `.env` and fill in your Entrez email and API key (optional but recommended for higher rate limits)

## 🚀 Usage
Modify `.env` and `radiology_dataset_db/config.py` as needed to customize PubMed query/LLM settings. Runtime defaults like modality/output paths are now configured via CLI args in `scripts/build_db.py`. Then run:
```bash
python scripts/build_db.py --database-modality MODALITY
```

## Currently supported modalities:
- Radiology: `--database-modality radiology`
- Single-cell RNA-seq: `--database-modality scrnaseq`
- Bulk genomics (e.g., bulk RNA-seq, WGS/WXS): `--database-modality bulk_genomics`
- Spatial transcriptomics: `--database-modality spatial_transcriptomics`

To enable parallel extraction, increase `--num-threads` (for example `--num-threads 8`).


## To add more modalities (e.g., genomics, pathology):
1. Define new dataset schema and extraction instructions in `radiology_dataset_db/config.py`
2. Implement new class and extraction function in `radiology_dataset_db/extract_MODALITY_dataset_information_llm.py`
3. Import and call the new extraction function in `scripts/build_db.py` and add a conditional to check the modality type
4. Optionally, update .github/workflows/update_dbs.yml to run the pipeline for the new modality on a schedule
5. Optionally, add some ground truth papers to tests/conftest.py, and add to get_modality_info in tests/test_llm_output.py to check that the new extraction function is working as expected
All instructions are notaded in the code with comments like `#* add additional extraction instructions and functions for other modalities here, e.g. genomics, pathology, etc`

Example codex prompt used to add a scRNA-seq dataset schema and extraction function:
```text
Pleas write a module very similar to extract_radiology_dataset_information_llm.py called extract_scrnaseq_dataset_information_llm.py that looks for scRNA-seq/snRNA-seq data. It should look for name, num_patients, sequencing_technology (eg 10X, SMARTSEQ, Parse, etc), disease, species, tissue, cell/nuclei. Also have fields for paper_title, paper_link, paper_year etc that get populated afterwards. Add instructions in config.py, and add an extra condition to build_db.py (areas to edit are marked by "#*"). Add integration test structure in test_llm_output.py and add a placeholderground truth paper to conftest.py to test the new extraction function.
```

## Testing
### Just unit tests:
`pytest`

### Just integration tests:
`pytest -m integration`
(not all tests need to pass because LLM has some randomness, but most should pass consistently)

### All tests:
`pytest -m ""`
