# Ноутбуки

Ноутбуки используются только как оболочка для запуска готового кода из `src/`.

Доступные шаблоны:

- `kaggle_teacher_pseudolabeling.ipynb` - Teacher pseudo-labeling.
- `kaggle_train_student.ipynb` - обучение Student baseline.
- `kaggle_full_pipeline.ipynb` - полный Kaggle pipeline от Teacher до Student evaluation.

Планируемый Kaggle notebook должен быть коротким:

```python
from src.training.train import train_student

train_student(
    config_path="configs/training/student.yaml",
    data_dir="/kaggle/input/...",
    output_dir="/kaggle/working/checkpoints",
)
```

Логика датасетов, моделей, метрик и обучения должна оставаться в репозитории.
