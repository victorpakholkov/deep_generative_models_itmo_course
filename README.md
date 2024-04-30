**Пахолков Виктор Владимирович**

**Курс "Глубокие генеративные модели (Deep Generative Models)" в AITalantedHub**


## ДЗ 4. Обучение Stable diffusion 1.5 методом Dreambooth

### 0. Data

Для выполнения данной домашней работы был создан датасет, состоящий из одного персонажа и 45 изображений.
Персонажем был выбран Доналд Трамп.
Были включены разные кадры с вариативностью, как более близкие, так и среднего плана, с различными поворотами и выражениями лица (благо трамп феноменально отличается от фотографии к фотографии и найти две похожие сложно).

Далее картинки были обработаны с помощью предложенного инструсмента: birme.

Cоотношение - 1 к 1, высота и широта - 512, автоматическая детекция focal point.

### 1. Stable diffusion 1.5.

Для выполнения первого пункта этого задания была обучена модель sd1.5.

Была использована библиотека diffusers и предобученная модель с сайта civitai, конвертированная в формат, прдходящий для diffusers.

Инстанс промпт был "a photo of sks man face", класс промпт - "a photo of man face".
Разрешение - 512, размер тренировочного батча - 1, learning rate - 2e-6 при 500 класс имаджес и 800 тренировочных шагах (с 8 битным адамом).

После обучения был проведен инференс с 2 сэмплами, 7.5 гайданс скейл, 35 (и впоследствии 30) шагами инференса

Результаты инференса модели:

![kitchen (1)](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/582c769b-5220-4f7a-b304-aa619220cc95)
*Рис.1 Трамп на кухне*

![anime (2)](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/b1651fc5-1c91-4300-8279-52cf91d9209a)
*Рис.2 Трамп в стиле аниме (как-то не совсем аниме)*

![classicist art](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/1723b553-b340-4fce-abeb-8c4bab523dc5)
*Рис.3 Классицистский трамп*

![forest (1)](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/f0e7fded-1e5d-4a43-9483-edbbb5f7002d)
*Рис.4 Трамп в лесу*



### 2. LoRA

Выбирал rank 4, 8 и 24, понизив lr до 5e-4

Lora rank 4:

![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/f48cd7d3-d4f0-41d5-bf59-0812e0bc82de)

Lora rank 8:
![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/6f10e838-a1c2-47b7-bbd5-53e81cdb4a3a)

Lora rank 24:

![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/2437f324-a1d8-4f4d-8639-da680c13bb29)


### 3. Сравнение Unet и Lora





### 4. ControlNet

