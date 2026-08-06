.PHONY: install dev test lint run smoke up down logs key

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

test:
	STORAGE_BACKEND=memory JOB_MODE=inline pytest -q

lint:
	ruff check corpusretrieve service tests tools

run:
	STORAGE_BACKEND=memory JOB_MODE=inline \
	python -m uvicorn service.app:app --reload --port 8000

smoke:
	python tools/api_smoke_test.py

key:
	@python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f api worker
