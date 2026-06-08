# Обучение Student-модели в Kaggle

Обучение Student-модели планируется запускать не локально, а в Kaggle notebook.
Локальный репозиторий при этом должен содержать всю основную логику:

- загрузка COCO-like датасета;
- модель;
- loss;
- train loop;
- метрики;
- сохранение checkpoints.

Notebook должен быть короткой оболочкой запуска.

## Что нужно подготовить до Kaggle

1. Полный датасет pedestrian crops.
2. Filtered crops после QA.
3. Teacher pseudo-labels.
4. COCO-like pose dataset.
5. Конфиг Student-модели и обучения.

Teacher inference тоже можно запускать в Kaggle/Colab GPU-среде. Для этого добавлен шаблон:

```text
notebooks/kaggle_teacher_pseudolabeling.ipynb
```

## Что загружать в Kaggle

Минимальный вариант:

```text
repo snapshot
data/pseudo/nuscenes_pose_coco/
```

Для Teacher pseudo-labeling сначала нужно загрузить filtered crops:

```text
data/exports/nuscenes_pedestrian_crops_filtered.zip
```

Этот архив можно получить локально:

```bash
.venv/bin/python scripts/export_dataset.py \
  --input-dir data/processed/nuscenes_pedestrian_crops_filtered \
  --output-path data/exports/nuscenes_pedestrian_crops_filtered.zip \
  --store
```

Если данные ведутся через DVC remote, в Kaggle можно сделать:

```bash
git clone <repo>
cd pose-estimation
pip install -r requirements-kaggle.txt
dvc pull data/pseudo/nuscenes_pose_coco
```

Если DVC remote еще не настроен, можно загрузить `data/pseudo/nuscenes_pose_coco` как Kaggle Dataset и подключить его к notebook.

## Идеальный notebook

Notebook должен выглядеть примерно так:

```python
from src.training.train import train_student

train_student(
    config_path="configs/training/student_baseline.json",
    data_dir="/kaggle/input/nuscenes-pose-coco",
    output_dir="/kaggle/working/checkpoints",
)
```

## Что пока не реализовано

Добавлен первый чистый PyTorch baseline:

```text
SimplePoseStudent
```

Это компактная heatmap-based CNN-модель без MMPose/MMCV. Она обучается на `annotations.json` в COCO-like формате и сохраняет:

```text
best.pt
last.pt
history.json
```

До полноценного сравнения еще нужно добавить:

1. speed benchmark;
2. экспорт модели;
3. визуализацию Student predictions;
4. более строгие PCK/OKS метрики.
