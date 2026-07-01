###############################################################################
#  BCG X Analytics Accelerator — NovaMart Retail Engagement
#  Production Makefile
###############################################################################

.DEFAULT_GOAL := help
SHELL := /bin/bash

# ─── Colours ──────────────────────────────────────────────────────────────────
BOLD   := \033[1m
CYAN   := \033[36m
GREEN  := \033[32m
YELLOW := \033[33m
RESET  := \033[0m

###############################################################################
# INSTALL
###############################################################################

.PHONY: install
install:  ## Install package and dev dependencies (editable)
	uv pip install -e ".[dev]"

###############################################################################
# CODE QUALITY
###############################################################################

.PHONY: lint
lint:  ## Run Ruff linter across all source trees
	ruff check src/ api/ dashboard/ tests/

.PHONY: format
format:  ## Run Ruff formatter across all source trees
	ruff format src/ api/ dashboard/ tests/

.PHONY: format-check
format-check:  ## Check formatting without modifying files (CI mode)
	ruff format --check src/ api/ dashboard/ tests/

.PHONY: typecheck
typecheck:  ## Run mypy type checker on the bcgx package
	mypy src/bcgx/

.PHONY: pre-commit-install
pre-commit-install:  ## Install pre-commit hooks
	pre-commit install

###############################################################################
# TESTS
###############################################################################

.PHONY: test
test:  ## Run unit tests only
	pytest tests/unit/ -v

.PHONY: test-integration
test-integration:  ## Run integration tests only
	pytest tests/integration/ -v

.PHONY: test-all
test-all:  ## Run the full test suite
	pytest -v

.PHONY: coverage
coverage:  ## Run tests and generate HTML coverage report
	pytest --cov=src/bcgx --cov-report=html
	@echo -e "$(GREEN)Coverage report generated at htmlcov/index.html$(RESET)"

###############################################################################
# DATA & PIPELINE
###############################################################################

.PHONY: generate-data
generate-data:  ## Generate synthetic NovaMart retail dataset
	python scripts/generate_data.py

.PHONY: run-audit
run-audit:  ## Run data quality audit
	python scripts/run_audit.py

.PHONY: train-models
train-models:  ## Train all ML models
	python scripts/train_models.py

###############################################################################
# SERVICES
###############################################################################

.PHONY: run-api
run-api:  ## Start FastAPI backend (development, hot-reload)
	uvicorn api.main:app --reload --port 8000

.PHONY: run-dashboard
run-dashboard:  ## Start Streamlit dashboard
	streamlit run dashboard/app.py

.PHONY: run-mlflow
run-mlflow:  ## Start MLflow tracking server
	mlflow ui --port 5000

###############################################################################
# DOCKER
###############################################################################

.PHONY: docker-build
docker-build:  ## Build all Docker images
	docker-compose build

.PHONY: docker-up
docker-up:  ## Start full stack in detached mode
	docker-compose up -d

.PHONY: docker-down
docker-down:  ## Stop and remove all containers
	docker-compose down

###############################################################################
# CLEAN
###############################################################################

.PHONY: clean
clean:  ## Remove all build artefacts, caches, and temp files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	find . -name "coverage.xml" -delete 2>/dev/null || true
	@echo -e "$(GREEN)Clean complete$(RESET)"

###############################################################################
# HELP
###############################################################################

.PHONY: help
help:  ## Show this help message
	@echo ""
	@echo -e "$(BOLD)$(CYAN)BCG X Analytics Accelerator — NovaMart Retail Engagement$(RESET)"
	@echo -e "$(CYAN)────────────────────────────────────────────────────────$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BOLD)%-25s$(RESET) %s\n", $$1, $$2}'
	@echo ""
