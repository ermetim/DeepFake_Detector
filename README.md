# Определение дипфейков по фотографии.
Годовой проект по первому курсу магистерской подготовки МОВС ВШЭ.

## **Команда: Тимур Ермешев.**

## **Куратор: Вячеслав Пирогов.**

### **Описание проекта:**

Дипфейк - методика синтеза изображений, основанная на искусственном интеллекте, которая используется для соединения и наложения существующих изображений на исходные изображения. 
Технология дипфейк используют для создания фальшивых фотографий со знаменитостями, а также для создания поддельных новостей.

Данный проект будет решать задачу классификации изображений и определять является ли изображение дипфейком или нет.


### **План работы:**

1. Первый этап.
- Создать репозиторий.
- Подобрать готовые датасеты для работы и/или подобрать методы для генерации датасета.
- Создании генератора новых изображений-дипфейков (опционально, возможно в 4-5 этапе).
- Произвести анализ фотографий из датасета.
- Подготовить скрипт по оценке фотографий на чтение, произвести очистку при необходимости.

2. Второй этап (ML)
- Подготовить функцию для перевода картинки в вектор.
- Произвести выбор модели для бинарной классификации картинок (дипфейк или нет).
- Произвести подбор гиперпараметров для выбранной модели для улучшения качества.

3. Третий этап (DL)
- Выбор архитектуры предобученной нейросети.
- Создать функцию для загрузки изображений.
- Создание функции для обучения и валидации модели и определение количества эпох.
- Создание функции для тестирования модели.

4. Четвертый этап (DL)
- Другие методы определения дипфейков.
- Трансформаторы

5. Пятый этап
- Создание веб-сервиса/ТГ бот, в который пользователь загружает фотографию и получает в ответ решение о том, является ли картинка дипфейком или нет.


### **Данные:**

1. Датасет [FaceForensics++](https://github.com/ondyari/FaceForensics/tree/master/dataset)
   - 3068 manipulated videos.
   - 363 original source actor videos.

2. Датасет [Celeb-DF](https://paperswithcode.com/dataset/celeb-df)
   - 590 original videos collected from YouTube
   - 5639 corresponding DeepFake videos

### Структура проекта

```
DeepFake_Detector
├── datasets
│   ├── frames
│   │   ├── test/
│   │   ├── train/
│   │   └── val/
│   ├── videos
│   │   ├── celebdf_2/
│   │   └── faceforensics/
├── models
│   ├── DL/
│   └── ML/
├── notebooks
│   ├── eda
│   │   └── EDA.ipynb
│   ├── train_models
│   │   ├── DL_model.ipynb
│   │   └── ML_model.ipynb
│   └── Get_pictures_from_video.ipynb
├── presentations/
├── src
│   ├── faceforensics_dataset_downloader
│   │   ├── faceforensics_download_v4.py
│   │   └── README.md
│   ├── dl_functions.py
│   └── ml_functions.py
├── tg_bot/
├── .env
├── .gitignore
├── dl_train.py
├── frames_from_video.py
├── ml_train.py
└── README.md
```

1. `datasets/`
   1. `frames/` - test, train and val frames with real and fake classes
   2. `videos/` - celebdf_2 and faceforensics++ video datasets
2. `models` - trained best ML and DL models
3. `notebooks/`
   1. `eda/EDA.ipynb` - notebook with exploratioin data analisys of the datasets.
   2. `train_models/` - notebooks with dataset preprocessing and training ML/DL models
   3. `Get_pictures_from_video.ipynb` - notebook for splitting datasets to train, test, val. Extracting frames from videos.
4. `presentations/` - checkpoint presentations
5. `src/`
   1. `faceforensics_dataset_downloader/` - faceforensics dataset downloader with readme file for running .py file.
   2. `dl_functions.py` - functions loading pictures, preprocessing and training DL models.
   3. `ml_functions.py` - functions loading pictures, preprocessing and training ML models.
6. `tg` - files, associated with tg bot
7. `.gitignore` - list with git ignore files
8. `.env` - environment secret constants
9. `dl_train.py` - download pictures, transform, train and save DL model
10. `frames_from_video.py` - splitting dataset, extracting frames from videos, save pictures
11. `ml_train.py` - download pictures, transform, train and save ML models
12. `README.md` - project description


