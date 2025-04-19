.PHONY: venv install data offline_insert online_query

VENV=.my_venv
PYTHON_VERSION=python3# Set or change python version if desired.
PYTHON=$(VENV)/bin/$(PYTHON_VERSION)

venv: # Create Virtual Environment
	$(PYTHON_VERSION) -m venv $(VENV)

install: # Install Dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --no-cache-dir -r requirements.txt
	$(PYTHON) -m ipykernel install --user --name=$(VENV)

DATA_GEN_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
NUM_ARTICLES=10
data: # Synthesize PII and non-PII data
	$(PYTHON) data_synthesizer.py --model_name $(DATA_GEN_MODEL) --num_articles $(NUM_ARTICLES)

EMBEDDING_MODEL=all-MiniLM-L6-v2
offline_insert: # Convert synthesized data into a vector database for RAG
	$(PYTHON) offline_db_insert.py --embedding_model $(EMBEDDING_MODEL)

MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
online_query: # Real-time query to assess claims
	$(PYTHON) online_query.py --model_name $(MODEL_NAME) --embedding_model $(EMBEDDING_MODEL)