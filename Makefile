SHELL := /usr/bin/env bash
PYTHON ?= python
PROJECT_ROOT := $(PWD)
ENV_FILE := .env

.PHONY: help setup install-dev pre-commit index api ui eval test lint format docs

help:
	@echo "Available targets:"
	@echo "  setup       Install runtime and dev dependencies"
	@echo "  pre-commit  Run pre-commit on all files"
	@echo "  index       Run the ingestion pipeline against /data"
	@echo "  api         Start the FastAPI server"
	@echo "  ui          Launch the Streamlit UI"
	@echo "  eval        Execute the evaluation harness"
	@echo "  test        Run pytest, mypy, and ruff"
	@echo "  lint        Run ruff and mypy"
	@echo "  format      Auto-format using black and ruff"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	pre-commit install

pre-commit:
	pre-commit run --all-files

index:
	$(PYTHON) -m src.ingestion.indexer --data-root data --index-dir .cache/index

api:
	env $(if $(wildcard $(ENV_FILE)), $$(cat $(ENV_FILE) | xargs)) uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	env $(if $(wildcard $(ENV_FILE)), $$(cat $(ENV_FILE) | xargs)) streamlit run ui/app.py

eval:
	$(PYTHON) -m src.evaluation.run_ragas --output docs/eval_report.md

lint:
	ruff check src tests
	mypy src tests

format:
	ruff format src tests
	black src tests

test:
	pytest -q
	mypy src tests
	ruff check src tests
