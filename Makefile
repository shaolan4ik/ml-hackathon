POETRY ?= poetry
PYTHON ?= python

.PHONY: install run test precommit migrate compose-up compose-build compose-down compose-logs

install:
	$(POETRY) install

run:
	$(POETRY) run $(PYTHON) -m hackaton.service.main

test:
	$(POETRY) run pytest

precommit:
	$(POETRY) run pre-commit run --all-files

migrate:
	$(POETRY) run $(PYTHON) -m hackaton.service.db migrate

compose-build:
	docker compose build

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f app

