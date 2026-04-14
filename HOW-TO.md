# HOW-TO для участников

## Цель документа

Этот файл помогает быстро понять:

- где в проекте вносить изменения;
- как локально проверить, что решение корректное;
- как интерпретировать итог оценки.

## Минимальный маршрут участника

### Поднять сервис

```bash
make install
make migrate
make run
```

### Проверить качество кода

```bash
make test
make precommit
```

### Проверить, что CI пройдет

- Убедитесь, что локально зеленые `make test` и `make precommit`.
- Проверьте, что ваш код не ломает runtime smoke-цепочку:
  - сервис стартует;
  - данные `user/shift/event` загружаются;
  - `prepare` завершается, `ready` возвращает готовность;
  - `predict` возвращает непустой список кандидатов.
- Для проверки нагрузочного контура перед пушем можно запустить:

```bash
make load-test
```

### Обучить baseline/свою модель

```bash
poetry run python -m hackaton.train.cli train \
  --user-path data/train/user.csv \
  --shift-path data/train/shift.csv \
  --event-path data/train/event.csv \
  --output-dir artifacts/train \
  --skip-shap
```

### Запустить eval

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

### Проверить отчет

- `artifacts/eval_run/eval_report.md`

## Как добавлять новую библиотеку

Используйте только `poetry`, чтобы зависимости и lock-файл оставались согласованными.

Для runtime-зависимости:

```bash
poetry add <package-name>
```

Для dev-зависимости:

```bash
poetry add --group dev <package-name>
```

После добавления:

```bash
make test
make precommit
```

Что важно:

- не редактируйте `poetry.lock` вручную;
- если меняется состав/версии библиотек, коммитите вместе и `pyproject.toml`, и `poetry.lock`.

## Куда вносить изменения

- `hackaton/service/app.py`
  - online-логика `predict`;
  - правила фильтрации/ранжирования кандидатов;
- `hackaton/train/training.py`
  - feature engineering;
  - выбор/настройка модели;
  - train-артефакты.

## Что лучше не трогать

- Контракты RPC-методов (`user`, `event`, `shift`, `prepare`, `ready`, `predict`).
- Контракты входных CSV.
- Ограничение `predict_max_rpm <= 200`.
- Фиксированный пул: `pool_size = 10`.
- Ограничение FPR: `max_fpr = min(1.0, capacity / 10)`.
- Агрегация:
  - shift -> capacity-group/day;
  - day -> overall metric.

Подробнее про регламент можно прочитать в: `REGLAMENT.md`.

## Как читать eval-отчет

В `eval_report.md` смотрите:

- `overall_target_metric` — итоговый score решения.
- `predict_latency_p50/p80/p95` — задержки запросов `predict`.
- `predict_rpm` — фактический темп запросов.
- `prepare_duration_*` — цена подготовки модели.
- `Daily metrics` — детализация качества по дням и группам `capacity`.

## Частые проблемы и что делать

- `predict` возвращает `503 model is in prepare state`
  - дождитесь `ready` ;
  - проверьте таймауты `prepare_*_timeout_sec`.
- Eval падает на лимите RPM
  - уменьшите `--predict-max-concurrency`;
  - держите `--predict-max-rpm` не выше 200.
- Пустая/нулевая метрика
  - проверьте, что в `apply.csv` есть валидные совпадения с `predict`;
  - проверьте корректность `user_id/shift_id/date`.
- Изменили версии в `pyproject.toml` вручную, и `poetry.lock` устарел
  - перегенерируйте lock: `poetry lock`;
  - установите зависимости из lock: `poetry install`;
  - если нужно обновить конкретный пакет до новой версии: `poetry add <package-name>@<version>`;
  - если lock поврежден или конфликтный, можно пересобрать полностью:
    - `rm poetry.lock`
    - `poetry lock`
    - `poetry install`

## Чеклист перед MR

- Используйте `CHECKLIST.md` как обязательный pre-commit/pre-MR список.
- Минимум перед отправкой:
  - `make test` зеленый;
  - `make precommit` зеленый;
  - CI-цепочка из `.github/workflows/ci.yml` не должна падать на smoke-check;
  - `eval_report.md` формируется и читается.
