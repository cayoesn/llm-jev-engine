.PHONY: install run test coverage lint docker-up docker-down docker-logs

PYTHON ?= .venv/bin/python

install:
	$(PYTHON) -m pip install -e ".[dev]"

run:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8016 --reload

test:
	$(PYTHON) -m pytest -v

coverage:
	$(PYTHON) -m pytest --cov=app --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check app tests

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f
