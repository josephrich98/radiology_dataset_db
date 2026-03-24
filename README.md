conda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm

git clone git@github.com:josephrich98/radiology_dataset_db.git
cd radiology_dataset_db
conda create -n radiology_dataset_db python=3.10 -y
conda activate radiology_dataset_db
pip install -r requirements.txt

pytest
pytest -m integration