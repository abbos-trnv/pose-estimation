# Выбор Teacher-модели

Teacher используется только для генерации псевдоразметки. Его скорость не является приоритетом.
Главное требование - качественная разметка keypoints на кропах пешеходов.

## Рекомендуемый первый вариант

Для первого baseline стоит использовать сильную top-down pose-модель через MMPose:

```text
ViTPose-H или ViTPose-G
```

Причина: семейство ViTPose показывает сильные результаты на COCO keypoints и хорошо подходит для задачи Teacher inference, где допустима высокая вычислительная стоимость.

## Практичный первый запуск

Если ViTPose-H/G окажется неудобным по весам, зависимостям или памяти GPU, практичный первый Teacher:

```text
RTMPose-l AIC+COCO 256x192
```

RTMPose проще использовать в экосистеме MMPose и удобнее для массовой разметки.
В MMPose model zoo для `rtmpose-l-aic-coco` указан COCO AP `0.765`, а для варианта `384x288` - `0.773`.
Это не самый тяжелый возможный Teacher, но хороший воспроизводимый первый шаг.

## Источники

- ViTPose paper: https://arxiv.org/abs/2204.12484
- ViTPose++ paper: https://arxiv.org/abs/2212.04246
- MMPose documentation/model zoo: https://mmpose.readthedocs.io/
- MMPose project: https://github.com/open-mmlab/mmpose

## Решение для проекта

Пока фиксируем направление:

1. Основной Teacher-кандидат по качеству: ViTPose-H.
2. Практичный Teacher для первого Kaggle-запуска: RTMPose-l AIC+COCO.
3. Схема keypoints: COCO-17.
4. Teacher запускается на pedestrian crops, а не на полном изображении.
5. После inference обязательно выполняется visual QA и фильтрация по confidence.

Для RTMPose-l AIC+COCO:

```text
config:
mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py

checkpoint:
https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth
```

В Kaggle удобнее скачать config и checkpoint автоматически:

```bash
mim download mmpose \
  --config rtmpose-l_8xb256-420e_aic-coco-256x192 \
  --dest /kaggle/working/mmpose_checkpoints
```

## Установка MMPose

Официальная документация MMPose рекомендует ставить `MMEngine` и `MMCV` через `MIM`.
Для Colab/Kaggle-подобной среды базовая схема такая:

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
pip install mmpose
```

Если ставить MMPose из исходников:

```bash
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -e .
```

## API inference

Для top-down inference MMPose предоставляет Python API:

```python
from mmpose.apis import init_model, inference_topdown

model = init_model(config, checkpoint, device="cuda:0")
result = inference_topdown(model, image_path, bboxes=bboxes, bbox_format="xyxy")
```

Именно эта схема используется в `src/teacher/mmpose_teacher.py`.
