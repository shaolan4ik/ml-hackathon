# DATA: форматы train/validation данных

## Назначение

Этот файл описывает входные CSV-контракты для:

- начальной загрузки (`train`);
- дневного цикла оценки (`validation`).

## Train (начальная загрузка)

Файлы:

- `data/train/user.csv`
- `data/train/shift.csv`
- `data/train/event.csv`

### `user.csv`

Колонки:

- `location_id` (string) — идентификатор локации пользователя.
- `is_strict_location` (bool) — пользователь явно выбрал локацию.
- `id` (string) — идентификатор пользователя.
- `has_mk` (bool) — наличие медкнижки.

Пример:

```csv
location_id,is_strict_location,id,has_mk
loc_1,true,u_1001,true
loc_2,false,u_1002,false
```

### `shift.csv`

Колонки:

- `id` (string)
- `start_at` (datetime string, ISO-8601)
- `location_id` (string)
- `task_type` (string)
- `employer_id` (string)
- `workplace_id` (string)
- `need_mk` (bool)
- `id_differential` (bool)
- `hours` (int)
- `reward` (float)
- `capacity` (int)

Пример:

```csv
id,start_at,location_id,task_type,employer_id,workplace_id,need_mk,id_differential,hours,reward,capacity
s_501,2026-03-24T08:00:00Z,loc_1,picker,e_10,w_77,true,false,8,1800.0,3
s_502,2026-03-24T10:00:00Z,loc_2,loader,e_12,w_90,false,false,6,1400.0,2
```

### `event.csv`

Колонки:

- `id` (uuid|string)
- `shift_id` (string)
- `user_id` (string)
- `interaction` (string): `VIEW`, `APPLY`, `FINISHED`, `USER_CANCEL`, `SYSTEM_CANCEL`
- `ts` (datetime string, ISO-8601)

Пример:

```csv
id,shift_id,user_id,interaction,ts
9f8f2ec9-b213-4f80-a2cb-66065a9e8cb3,s_501,u_1001,VIEW,2026-03-23T15:00:00Z
4d04eec1-3ccb-4fd8-bbd5-43e535d18ef6,s_501,u_1001,APPLY,2026-03-23T15:05:00Z
```

## Validation (дневной цикл оценки)

Файлы:

- `data/validation/apply.csv`
- `data/validation/shift.csv`
- `data/validation/event.csv`

### `apply.csv`

Колонки:

- `user_id` (string)
- `shift_id` (string)
- `date` (date, `YYYY-MM-DD`)

Пример:

```csv
user_id,shift_id,date
u_1001,s_601,2026-03-25
u_1002,s_602,2026-03-25
```

### `validation/shift.csv`

Схема такая же, как у `train/shift.csv`.

Пример:

```csv
id,start_at,location_id,task_type,employer_id,workplace_id,need_mk,id_differential,hours,reward,capacity
s_601,2026-03-25T09:00:00Z,loc_1,picker,e_10,w_77,true,false,8,1900.0,3
```

### `validation/event.csv`

Схема такая же, как у `train/event.csv`.

Пример:

```csv
id,shift_id,user_id,interaction,ts
f9d02b6f-5ce7-4574-80e3-366f6c1b4efa,s_601,u_1001,VIEW,2026-03-25T11:00:00Z
```

## Важные замечания

- Полный регламент расчета метрики и дневного цикла: `REGLAMENT.md`.
- Практический маршрут участника: `HOW-TO.md`.
