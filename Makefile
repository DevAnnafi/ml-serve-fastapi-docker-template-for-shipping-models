.PHONY: install dev test lint build up down clean

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --tb=short

lint:
	ruff check app/ tests/

build:
	docker build -t ml-serve:latest .

up:
	docker compose up --build

down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete; \
	rm -rf .pytest_cache .ruff_cache
