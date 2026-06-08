# Быстрая модель pose estimation пешеходов через псевдоразметку

Проект строит воспроизводимый пайплайн для обучения легкой модели pose estimation пешеходов в сценах автономного транспорта.
В качестве исходного датасета используется nuScenes: в нем есть bbox пешеходов, но нет разметки keypoints.

Основная идея: сильная Teacher-модель размечает keypoints на кропах пешеходов, после чего компактная Student-модель обучается на этой псевдоразметке.

## Цель

Получить маленькую и быструю модель pose estimation, которая работает на кропе пешехода:

```text
изображение -> bbox пешехода -> crop -> Student -> keypoints
```

На текущем этапе bbox считается известным. Детекция объектов не входит в baseline проекта.

## Что не входит в baseline

- object detection;
- pedestrian tracking;
- trajectory prediction;
- intention prediction;
- BEV-представления;
- 3D detection;
- sensor fusion;
- управление автомобилем.

Эти направления могут быть добавлены позже, но сейчас проект сфокусирован на pose estimation и подготовке псевдоразмеченного датасета.

## Pipeline

```text
nuScenes raw data
    ↓
извлечение bbox пешеходов
    ↓
pedestrian crops
    ↓
визуальная проверка кропов
    ↓
Teacher pose inference
    ↓
визуальная проверка keypoints
    ↓
фильтрация pseudo-labels
    ↓
COCO-like pose dataset
    ↓
обучение Student-модели в Kaggle
    ↓
оценка качества и скорости
```

## Текущий статус

- [x] Создан базовый репозиторий.
- [x] Настроены Python, Git и DVC.
- [x] Загружен nuScenes trainval.
- [x] Подготовлен `data/raw/nuscenes` в формате, который ожидает `nuscenes-devkit`.
- [x] Реализовано извлечение pedestrian crops из bbox nuScenes.
- [x] Выполнен smoke test: 29 кропов на первых 10 samples.
- [x] Добавлен QA-скрипт для визуальной проверки кропов и bbox.
- [ ] Проведена полная визуальная проверка кропов.
- [ ] Подключена сильная Teacher-модель.
- [ ] Сгенерированы keypoints pseudo-labels.
- [ ] Добавлена фильтрация pseudo-labels.
- [ ] Собран COCO-like датасет.
- [ ] Реализовано обучение Student-модели.
- [ ] Подготовлен короткий Kaggle notebook для запуска обучения.

## Структура репозитория

```text
configs/       # конфиги данных, Teacher, Student и обучения
data/
  raw/         # исходные данные, не хранятся в Git
  processed/   # кропы и промежуточные датасеты, не хранятся в Git
  pseudo/      # pseudo-labels, не хранятся в Git
  qa/          # визуальные QA-примеры, не хранятся в Git
logs/          # логи экспериментов, не хранятся в Git
models/        # checkpoints/weights, не хранятся в Git
notebooks/     # короткие notebook-обертки для Kaggle/анализа
scripts/       # CLI-скрипты пайплайна
src/           # основная логика проекта
dvc.yaml       # воспроизводимые DVC stages
```

В Git должны попадать код, конфиги, документация и DVC metadata. Сырые данные, кропы, pseudo-labels, checkpoints и логи не должны попадать в Git.

## Установка

```bash
uv sync
```

Если запускаете команды без `uv`, используйте Python из виртуального окружения:

```bash
.venv/bin/python --version
```

## Подготовка nuScenes

Если архивы trainval распакованы отдельными папками вида `*_meta` и `*_keyframes`, соберите devkit-compatible dataroot через symlinks:

```bash
.venv/bin/python scripts/prepare_nuscenes_dataroot.py \
  --raw-dir data/raw \
  --output-dir data/raw/nuscenes \
  --version v1.0-trainval
```

Ожидаемая структура:

```text
data/raw/nuscenes/
  samples/
  maps/
  v1.0-trainval/
```

Symlinks используются для того, чтобы не копировать десятки гигабайт данных.

Проверить, какая часть camera keyframes реально присутствует локально:

```bash
.venv/bin/python scripts/validate_nuscenes_files.py \
  --dataroot data/raw/nuscenes \
  --version v1.0-trainval \
  --output-path data/qa/nuscenes_file_coverage.json
```

## Экспорт датасета для Kaggle

Filtered crops занимают около 518 MB и могут быть загружены в Kaggle как отдельный Dataset.

```bash
.venv/bin/python scripts/export_dataset.py \
  --input-dir data/processed/nuscenes_pedestrian_crops_filtered \
  --output-path data/exports/nuscenes_pedestrian_crops_filtered.zip \
  --store
```

Архив содержит папку `nuscenes_pedestrian_crops_filtered/` с `crops/`, `manifest.jsonl`, `manifest.json`, `rejected.jsonl` и `report.json`.

## Извлечение кропов пешеходов

Smoke test:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/extract_nuscenes_pedestrian_crops.py \
  --dataroot data/raw/nuscenes \
  --version v1.0-trainval \
  --output-dir data/processed/nuscenes_pedestrian_crops_smoke \
  --limit-samples 10
```

Полный trainval:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/extract_nuscenes_pedestrian_crops.py \
  --dataroot data/raw/nuscenes \
  --version v1.0-trainval \
  --output-dir data/processed/nuscenes_pedestrian_crops
```

Результат:

```text
data/processed/nuscenes_pedestrian_crops/
  crops/
  manifest.jsonl
  manifest.json
```

Каждая запись `manifest.jsonl` содержит:

- путь к кропу;
- путь к исходному camera image;
- `sample_token`;
- `sample_data_token`;
- `annotation_token`;
- camera channel;
- категорию пешехода;
- bbox в форматах `xyxy` и `xywh`;
- размер кропа;
- `visibility_token`.

## Визуальная проверка кропов

Перед запуском Teacher-модели нужно глазами проверить, что bbox и crop корректны.

Для smoke-датасета:

```bash
.venv/bin/python scripts/visualize_crops.py \
  --dataset-dir data/processed/nuscenes_pedestrian_crops_smoke \
  --dataroot data/raw/nuscenes \
  --output-dir data/qa/crops_preview_smoke \
  --num-samples 29
```

Для полного датасета:

```bash
.venv/bin/python scripts/visualize_crops.py \
  --dataset-dir data/processed/nuscenes_pedestrian_crops \
  --dataroot data/raw/nuscenes \
  --output-dir data/qa/crops_preview \
  --num-samples 64
```

Результат:

```text
data/qa/crops_preview/
  crop_grid.jpg
  report.json
  bbox_on_source/
```

`crop_grid.jpg` показывает сетку кропов. `bbox_on_source/` показывает bbox на исходных кадрах. `report.json` содержит базовую статистику по размерам кропов, камерам и visibility.

## Фильтрация кропов

Перед Teacher inference нужно убрать слишком маленькие или явно неподходящие кропы.
Пороговые значения лежат в `configs/filtering/crops_default.json`.

Smoke-фильтрация:

```bash
.venv/bin/python scripts/filter_crops.py \
  --input-dir data/processed/nuscenes_pedestrian_crops_smoke \
  --output-dir data/processed/nuscenes_pedestrian_crops_smoke_filtered \
  --config configs/filtering/crops_default.json \
  --copy-crops
```

Полный датасет:

```bash
.venv/bin/python scripts/filter_crops.py \
  --input-dir data/processed/nuscenes_pedestrian_crops \
  --output-dir data/processed/nuscenes_pedestrian_crops_filtered \
  --config configs/filtering/crops_default.json \
  --copy-crops
```

Результат:

```text
data/processed/nuscenes_pedestrian_crops_filtered/
  crops/
  manifest.jsonl
  manifest.json
  rejected.jsonl
  report.json
```

После фильтрации стоит снова построить QA preview уже для filtered dataset.

## Работа с DVC

Основные стадии описаны в `dvc.yaml`:

```bash
.venv/bin/dvc repro prepare_nuscenes_dataroot
.venv/bin/dvc repro extract_pedestrian_crops
.venv/bin/dvc repro filter_pedestrian_crops
.venv/bin/dvc repro qa_crops_preview
.venv/bin/dvc repro qa_filtered_crops_preview
```

Для быстрой проверки на первых 10 samples:

```bash
.venv/bin/dvc repro qa_filtered_crops_preview_smoke
```

Тяжелые данные не коммитятся в Git. Для полноценной командной работы нужно настроить DVC remote, например S3, Google Drive, SSH или локальное хранилище.

После настройки remote типичный сценарий будет таким:

```bash
dvc add data/raw/nuscenes
dvc add data/processed/nuscenes_pedestrian_crops
dvc add data/pseudo/nuscenes_pose
dvc push
```

В Git при этом попадают только `.dvc` metadata и код.

Примечание: `nuscenes-devkit` при инициализации загружает таблицы всего `v1.0-trainval`, поэтому даже smoke-запуск на первых samples может занимать заметное время.

## Teacher-модель

Teacher используется только для генерации псевдоразметки. Скорость Teacher не важна, важно качество keypoints.

Планируемый подход:

- использовать сильную pose-модель из актуальных бенчмарков;
- запускать Teacher на pedestrian crops;
- сохранять keypoints, confidence scores и служебные признаки качества;
- визуально проверять примеры разметки;
- фильтровать плохие pseudo-labels.

Кандидаты для первого baseline:

- ViTPose-H / ViTPose-G;
- HRNet-W48;
- RTMPose-x через MMPose.

Окончательный выбор Teacher нужно делать перед реализацией inference-слоя, с учетом доступности весов, лицензии, удобства запуска и качества на COCO keypoints.

Черновой запуск Teacher inference после установки MMPose и подготовки весов:

```bash
.venv/bin/python scripts/run_teacher_inference.py \
  --dataset-dir data/processed/nuscenes_pedestrian_crops_filtered \
  --output-dir data/pseudo/nuscenes_pose_teacher \
  --mmpose-config <path-or-config-name> \
  --checkpoint <path-to-checkpoint> \
  --device cuda:0
```

После этого нужно построить preview keypoints:

```bash
.venv/bin/python scripts/visualize_keypoints.py \
  --dataset-dir data/processed/nuscenes_pedestrian_crops_filtered \
  --labels-path data/pseudo/nuscenes_pose_teacher/pseudo_labels.jsonl \
  --output-dir data/qa/keypoints_preview
```

Если MMPose/MMCV не ставится в Kaggle, используйте HuggingFace ViTPose:

```bash
python scripts/run_hf_vitpose_teacher.py \
  --dataset-dir data/processed/nuscenes_pedestrian_crops_filtered \
  --output-dir data/pseudo/nuscenes_pose_teacher \
  --model-name usyd-community/vitpose-plus-base \
  --device cuda:0 \
  --limit 128
```

После Teacher inference используется умеренная фильтрация pseudo-labels:

```bash
python scripts/filter_pseudo_labels.py \
  --labels-path data/pseudo/nuscenes_pose_teacher/pseudo_labels.jsonl \
  --output-dir data/pseudo/nuscenes_pose_teacher_filtered \
  --config configs/filtering/pseudo_labels_default.json
```

Baseline-правило: сохранять примеры с mean confidence `>= 0.5` и минимум `8` уверенными keypoints.

И собрать COCO-like dataset:

```bash
.venv/bin/python scripts/build_coco_pose_dataset.py \
  --dataset-dir data/processed/nuscenes_pedestrian_crops_filtered \
  --labels-path data/pseudo/nuscenes_pose_teacher/pseudo_labels.jsonl \
  --output-dir data/pseudo/nuscenes_pose_coco \
  --copy-images
```

Для проверки downstream-пайплайна без реального Teacher можно сгенерировать синтетические keypoints.
Их нельзя использовать для обучения:

```bash
.venv/bin/python scripts/generate_dummy_pseudo_labels.py \
  --dataset-dir data/processed/nuscenes_pedestrian_crops_smoke_filtered \
  --output-dir data/pseudo/dummy_pose_smoke \
  --limit 16
```

## Student-модель

Student - финальная модель проекта. Она должна быть компактной и быстрой.

Кандидаты:

- LiteHRNet;
- MobileNet-based heatmap pose model;
- RTMPose-t / RTMPose-s;
- другая легкая heatmap-based архитектура.

Обучение планируется запускать в Kaggle notebook. При этом вся логика датасета, модели, loss, train loop и метрик должна жить в `src/`, а notebook должен быть короткой оболочкой запуска.

Первый baseline реализован без MMPose/MMCV:

```text
SimplePoseStudent
```

Это компактная heatmap-based CNN-модель. Запуск обучения:

```bash
python scripts/train_student.py \
  --config configs/training/student_baseline.json \
  --data-dir data/pseudo/nuscenes_pose_coco \
  --output-dir models/student_baseline
```

Результат:

```text
models/student_baseline/
  best.pt
  last.pt
  history.json
```

## Метрики

Качество:

- PCK;
- OKS.

Производительность:

- latency, мс на crop;
- FPS;
- размер модели;
- число параметров.

## Ближайшие шаги

1. Прогнать извлечение кропов на всем trainval.
2. Сгенерировать QA-превью кропов.
3. Глазами проверить ошибки bbox/crop.
4. Определить правила фильтрации кропов.
5. Выбрать Teacher-модель.
6. Реализовать Teacher inference.
7. Сгенерировать и проверить keypoints preview.

## Отчеты

Подробный ход работ ведется в:

- `docs/PROGRESS.md`;
- `docs/TEACHER_SELECTION.md`;
- `docs/KAGGLE_TRAINING.md`.
