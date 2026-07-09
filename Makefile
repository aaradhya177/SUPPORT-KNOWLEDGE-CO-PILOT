.PHONY: install lint format test ingest build-indexes run-api run-frontend eval docker-up

install:
	pip install -r requirements.txt

lint:
	ruff check .

format:
	black .
	ruff check . --fix

test:
	pytest

ingest:
	python scripts/run_ingestion.py

build-indexes:
	python scripts/build_indexes.py

run-api:
	uvicorn app.main:app --reload

run-frontend:
	streamlit run frontend/streamlit_app.py

eval:
	python eval/run_eval.py

docker-up:
	docker compose up --build
