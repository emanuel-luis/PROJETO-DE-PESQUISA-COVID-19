.DEFAULT_GOAL := help
.PHONY: help install lint test test-fast pipeline demo clean-demo

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Install locked dependencies
	uv sync --locked

lint: ## Ruff check, ruff format check and mypy
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

test: ## Run tests, including slow ones
	uv run pytest -m "not external and not gpu"

test-fast: ## Run the default fast tests
	uv run pytest

pipeline: ## Reproduce the DVC pipeline and refresh the README results
	uv run dvc repro
	uv run covidxr report

demo: ## Run the whole pipeline on synthetic images in .demo
	uv run covidxr run-all --params params.ci.yaml --root .demo

clean-demo: ## Remove the demo output
	rm -rf .demo
