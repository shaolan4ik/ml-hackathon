# Baseline обучение

## Назначение

Модуль `hackaton/training` реализует baseline-пайплайн обучения для задачи предсказания выхода пользователя на смену.

## Входные данные

Используются три файла из `data/train`:

- `user.csv`
- `shift.csv`
- `event.csv`

Перед обучением выполняется fail-fast валидация схем по контракту из `hakaton.md`.

## Что делает baseline

1. Загружает и валидирует train CSV.
2. Формирует пары `user-shift` и целевую переменную:
   - `target=1`, если для пары есть `APPLY` или `FINISHED`;
   - `target=0` для остальных наблюдаемых пар.
3. Исключает утечку по времени: события после `shift.start_at` не используются.
4. Строит признаки:
   - пользовательские (`has_mk`, `is_strict_location`, исторические счетчики);
   - сменные (`hours`, `reward`, `need_mk`, `id_differential`, `capacity`, `task_type`);
   - парные (`location_match`, `need_mk_match`, статистики завершенных смен по работодателю/точке).
5. Делит данные по времени в пропорции ~80/20.
6. Обучает `LogisticRegression`.
7. Считает целевую метрику по регламенту:
   - TOP-K кандидатов по `capacity` смены;
   - `ROC-AUC` с ограничением FPR по емкости;
   - агрегация по группам емкости и дням.
8. Сохраняет артефакты и SHAP-графики важности признаков.

## Запуск

```bash
poetry run python -m hackaton.train.cli train \
  --user-path data/train/user.csv \
  --shift-path data/train/shift.csv \
  --event-path data/train/event.csv \
  --output-dir artifacts/train
```

Опционально можно пропустить SHAP:

```bash
poetry run python -m hackaton.train.cli train \
  --user-path data/train/user.csv \
  --shift-path data/train/shift.csv \
  --event-path data/train/event.csv \
  --output-dir artifacts/train \
  --skip-shap
```

## Артефакты

В `output-dir` формируются:

- `model.pkl`
- `metrics.json`
- `feature_schema.json`
- `train_config.json`
- `data_contract_check.json`
- `train_report.md`
- `plots/shap_summary.png`, `plots/shap_bar.png` (или `plots/shap_skipped.txt` при ошибке SHAP)

## Переиспользуемый модуль метрики

Целевая метрика вынесена в отдельный модуль:

- `hackaton/eval/metric.py`
