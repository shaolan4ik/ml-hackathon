# Базовое решение для ML-хакатона

Репозиторий содержит baseline-реализацию сервиса под задачу ранжирования кандидатов на смены:

- API-сервис на `zero` (TCP RPC);
- baseline train-пайплайн;
- eval-пайплайн с дневной симуляцией и расчетом целевой метрики;
- инфраструктура проверки качества (tests, coverage gate, pre-commit, load-test).

## Общий процесс участия

Участники проходят следующий сценарий:

1. Получают доступ к репозиторию и клонируют кодовую базу.
1. Изучают контракты API, инструкции запуска и автотесты.
1. Получают train/validation данные в CSV-формате.
1. Обучают собственную модель и готовят решение.
1. Интегрируют модель/алгоритмы в сервис, не нарушая контракты.
1. Запускают тесты, eval и нагрузочные проверки.
1. Оформляют Merge Request в своей ветке.
1. После дедлайна организаторы фиксируют изменения и оценивают качество.

## Навигация по документам

- `README.md` — общий обзор, quickstart и команды.
- `HOW-TO.md` — практический маршрут участника (куда и как вносить изменения).
- `CHECKLIST.md` — checklist перед коммитом и MR.
- `DATA.md` — форматы train/validation таблиц и примеры.
- `REGLAMENT.md` — регламент валидации, метрика и запуск eval.
- `TRAIN.md` — детали train-пайплайна и train-артефактов.

## Быстрый старт (time-to-first-run)

```bash
make install
make migrate
make run
```

В отдельном терминале:

```bash
make test
```

## Где что лежит

- `hackaton/service` — RPC-сервис, DTO, работа с БД, `prepare/ready/predict`.
- `hackaton/train` — baseline обучение и сохранение train-артефактов.
- `hackaton/eval` — дневной цикл оценки, расчет метрики, генерация `eval_report.md`.
- `tests/unit`, `tests/e2e` — unit/e2e тесты.
- `scripts/load_test.py` — нагрузочное тестирование `predict`.
- `REGLAMENT.md` — официальный регламент оценки.
- `DATA.md` — контракты данных с примерами.
- `TRAIN.md` — детали train-модуля.
- `HOW-TO.md` — практический гайд для участников (куда и как вносить изменения).
- `CHECKLIST.md` — чеклист перед коммитом решения.

## Запуск обучения

```bash
poetry run python -m hackaton.train.cli train \
  --user-path data/train/user.csv \
  --shift-path data/train/shift.csv \
  --event-path data/train/event.csv \
  --output-dir artifacts/train \
  --skip-shap
```

## Запуск оценки

```bash
poetry run python -m hackaton.eval.cli run \
  --host 127.0.0.1 \
  --port 8000 \
  --user-path data/train/user.csv \
  --shift-path data/train/shift.csv \
  --event-path data/train/event.csv \
  --val-apply-path data/validation/apply.csv \
  --val-shift-path data/validation/shift.csv \
  --val-event-path data/validation/event.csv \
  --output-dir artifacts/eval_run \
  --predict-max-concurrency 4 \
  --predict-max-rpm 200
```

Артефакт оценки: `artifacts/eval_run/eval_report.md`.

## Целевая метрика (кратко)

- Пул кандидатов для метрики фиксирован: `pool_size = 10`.
- Ограничение FPR: `max_fpr = min(1.0, capacity / 10)`.
- Агрегация: по группам емкости внутри дня, затем среднее по дням.

Полные правила: `REGLAMENT.md`.

Форматы данных и примеры: `DATA.md`.

## Качество и проверка

- `make test` — unit/e2e + coverage gate (`>=80%`).
- `make precommit` — ruff + hooks + pytest.
- `make load-test` — нагрузочный тест `predict` с markdown-отчетом.
- `make compose-up` — запуск в Docker.

## CI в GitHub

В репозитории настроены workflow:

- `.github/workflows/ci.yml`:
  - lint + pytest + coverage;
  - запуск сервиса;
  - RPC smoke-check (`user/shift/event -> prepare/ready -> predict`), где проверяется, что для смены подбираются исполнители.
- `.github/workflows/load-test.yml`:
  - ручной и nightly запуск нагрузочного сценария;
  - проверка budget (`failed_calls == 0`, `p95 <= 1000ms`);
  - публикация markdown-отчета нагрузочного теста как artifact.

Merge должен проходить только при зеленых CI-проверках.
