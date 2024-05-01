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

Далее, с помощью train_dreambooth_lora была обучена lora модель на основе обученной до этого dreambooth sd. 

Большинство параметров обучения были оставлены такими же, что и при обучении sd, но lr был понижен до 5e-4.

Были обучены 3 лоры с rank в 4, 8 и 24.

Как я понял, выбор rank является компромиссом между способностью модели адаптироваться к новым задачам и ее склонностью к переобучению. 
Большее значение означает, что модель имеет способность адаптироваться к новым задачам лучше, но риск переобучения возрастает. 
С другой стороны, меньшее значение снижает этот риск, но может ограничить способность модели адаптироваться к новым задачам.

После этого, с помощью safetensors.torch.load_file() и torch.save() веса лоры были конвертированы из safetensors в bin и определена функция инференса.

Вначале функция загружает предобученную модель и переводит ее на cuda, потом загружает веса lora.
Далее, функция генерирует изображения для каждого промпта из списка promt_list. Для каждого промпта функция создает генератор с заданным сидом, выполняет инференс с помощью модели lora и сохраняет сгенерированные изображения в папке.
При выполнении инференса функция использует параметры, такие как высота и ширина изображения, отрицательный промпт, число изображений на промпт, число шагов инференса, коэффициент направления и генератор, для настройки процесса генерации изображений.

Результаты инференса лоры с разными rank на примере Трампа в лесу:


Lora rank 4:

![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/f48cd7d3-d4f0-41d5-bf59-0812e0bc82de)

Lora rank 8:

![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/6f10e838-a1c2-47b7-bbd5-53e81cdb4a3a)

Lora rank 24:

![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/2437f324-a1d8-4f4d-8639-da680c13bb29)


### 3. Сравнение Unet и Lora


Prompt             |  Unet          | Lora rank 24 | 
:-------------------------:|:-------------------------:|:-------------------------:
Classic  |  ![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/067997c6-ec7e-4c9f-a170-edbf911fb5b2) | ![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/09a7c97a-e2d5-40b5-98f4-561040bacb3c)
Night city | ![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/db274ca0-c6a6-4df2-905f-7ea8b99be796) | ![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/eba77f87-d8a8-43e0-80e9-dad5829ce535)
Kitchen | ![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/2763e6d0-58a5-48a9-bdfe-3c692f54e7da) | ![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/c8e7dd15-0b2e-4df3-a197-400b1eff4003)
Forest | ![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/3a19f955-e1fd-4edd-b7ea-3f9801ee05d7) | ![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/bf8e7488-0f09-42e1-b3c5-05d61afd8253)
Anime | ![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/169a2eb3-aebb-42d8-97d7-300721503d24) | ![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/c06001b3-ad49-4d52-a030-7dbd7d78c5e2)


### 4. ControlNet

