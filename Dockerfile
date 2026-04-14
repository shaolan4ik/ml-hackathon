FROM python:3.12-slim

WORKDIR /app

ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_NO_INTERACTION=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock README.md /app/
RUN poetry install --only main --no-root --no-ansi

COPY . /app

EXPOSE 8000

CMD ["python", "-m", "hackaton.service.main"]
