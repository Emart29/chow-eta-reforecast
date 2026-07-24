# Convenience targets. On Windows, run the underlying commands directly
# (e.g. `python -m src.pipeline`) if `make` is not available.

PYTHON ?= python

.PHONY: install pipeline data serve dashboard test lint docker-build docker-up

install:
	$(PYTHON) -m pip install -r requirements.txt

pipeline:              ## Run the full pipeline: data -> models -> reports -> bundle
	$(PYTHON) -m src.pipeline

data:                  ## Simulate orders and the lifecycle event stream
	$(PYTHON) -m src.simulate.lifecycle

serve:                 ## Run the serving API (requires a built bundle)
	uvicorn api.main:app --reload

dashboard:             ## Run the Streamlit monitoring dashboard
	streamlit run dashboard/app.py

test:
	PYTHONPATH=. pytest -q

lint:
	ruff check .

docker-build:
	docker build -t chow-eta-reforecast .

docker-up:
	docker compose up --build
