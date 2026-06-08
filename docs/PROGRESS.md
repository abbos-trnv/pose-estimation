# Отчет о ходе работ

## Цель

Построить воспроизводимый pipeline для обучения легкой модели pose estimation пешеходов:

```text
nuScenes bbox -> pedestrian crop -> Teacher keypoints -> filtered pseudo-labels -> Student training
```

Обучение Student-модели планируется запускать в Kaggle. Локальный репозиторий должен содержать всю основную логику, а notebook должен быть короткой оболочкой запуска.

## Уже сделано

1. Подготовлена структура проекта.
2. Загруженный nuScenes trainval найден в `data/raw`.
3. Разрозненные папки nuScenes собраны в `data/raw/nuscenes` через symlinks.
4. Добавлен extraction pipeline для pedestrian crops.
5. Выполнен smoke test на первых 10 samples: получено 29 кропов.
6. Добавлен QA pipeline для визуальной проверки кропов:
   - `crop_grid.jpg`;
   - bbox на исходных кадрах;
   - `report.json` со статистикой.
7. Настроен Git hygiene: тяжелые данные не попадают в Git.
8. Добавлены DVC stages для smoke и full pipeline.
9. Исправлен DVC `site_cache_dir`, чтобы DVC работал в текущем окружении.
10. Добавлен слой фильтрации кропов перед Teacher inference.
11. Проверена фильтрация на smoke dataset: оставлено 28 из 29 кропов.
12. Добавлен Teacher wrapper для MMPose inference.
13. Добавлен visual QA для keypoints.
14. Добавлен конвертер Teacher pseudo-labels в COCO-like формат.
15. Добавлена документация по Kaggle training boundary.
16. Прогнан full extraction по `v1.0-trainval` metadata с доступными локальными camera images.
17. Получено `112021` raw pedestrian crops.
18. После фильтрации осталось `37622` кропа.
19. Сгенерирован full QA preview для filtered crops.
20. Проверено покрытие локальных camera keyframes: доступно `81174` из `204894` изображений, то есть около `39.6%`.
21. Проверен downstream-формат через dummy keypoints: keypoint preview и COCO-like converter работают.
22. Добавлен notebook-шаблон для Teacher pseudo-labeling в GPU-среде.
23. Добавлен экспорт filtered crops в Kaggle-friendly zip archive.
24. Создан архив `data/exports/nuscenes_pedestrian_crops_filtered.zip`: `458M`, `37626` файлов.
25. Добавлен HuggingFace ViTPose Teacher fallback для Kaggle без MMPose/MMCV.
26. Зафиксирована стратегия анализа pseudo-labels и добавлен фильтр Teacher confidence.
27. Добавлен первый чистый PyTorch Student baseline: heatmap-based CNN, dataset loader, weighted MSE loss, PCK validation и checkpoints.
28. Добавлены визуализация Student predictions и speed benchmark latency/FPS.

## Текущее наблюдение

Smoke preview показывает, что кропы извлекаются корректно, но часть примеров маленькая или размытая.
Перед Teacher inference нужна фильтрация по размеру, aspect ratio и visibility.

Full preview filtered crops показывает рабочий датасет, но в нем ожидаемо остаются сложные случаи:
размытие, частичные окклюзии, маленькие пешеходы, плотный городской контекст.
Эти случаи не стоит полностью удалять до Teacher, но после Teacher inference нужно фильтровать по keypoint confidence.

Локально сейчас загружена не вся camera-часть trainval:

```text
available keyframe camera images: 81174 / 204894
present fraction: 0.3962
```

Поэтому текущий full pipeline фактически построен по доступной части trainval.

## Текущий этап

Этап crops/filter/QA по доступной части trainval готов. Следующий этап - Teacher pseudo-labeling.
Кодовый интерфейс добавлен, но MMPose и веса Teacher пока не установлены.

## Следующие шаги

1. Загрузить `data/exports/nuscenes_pedestrian_crops_filtered.zip` в Kaggle Dataset.
2. Запустить `notebooks/kaggle_teacher_pseudolabeling.ipynb`.
3. Сгенерировать pseudo-labels сначала на небольшом filtered subset.
4. Проверить keypoints preview.
5. Отрегулировать фильтрацию по Teacher confidence.
6. Сгенерировать pseudo-labels на всем filtered dataset.
7. Собрать финальный COCO-like dataset.
8. Запустить визуализацию Student predictions и speed benchmark в Kaggle.
9. Зафиксировать первые итоговые метрики baseline.
