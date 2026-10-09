install:
	pip install -e ".[dev,pg]"

test:
	pytest

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests
	ruff check --fix src tests

demo:
	earningsiq demo

serve:
	uvicorn earningsiq.api:app --reload

db:
	docker compose up -d db

.PHONY: install test lint format demo serve db
