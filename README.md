**Пахолков Виктор Владимирович**

**Курс "Глубокие генеративные модели (Deep Generative Models)" в AITalantedHub**


## ДЗ 3. Sampling в латентном пространстве StyleGAN

### 0. Data

Для этого дз были выбраны 5 персонажей (Трамп, Байден, Харрис, Пенс, Джонсон), на каждого были загружены 4-5 изображений (код в ноутбуке).

### 1. Проекции изображений в пространстве StyleGAN

Изображения были кадрированы для соответствия данным из трейна с помощью блока Align images из ноутбука.

После этого были найдены проекции изображений в пространстве StyleGAN, методами из предоставленного ноутбука.

![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/f4cbe0be-9b17-4c11-ac35-1ba726ec6af0)

*Таблица 1. Проекции изображений в пространстве StyleGAN*

Используется два метода для проецирования изображений в пространство StyleGAN: project_image и project_image_e4e.

В методе project_image используется оптимизация для нахождения латентого вектора изображения. Сначала изображение загружается и преобразуется в тензор PyTorch. Затем создаются экземпляры классов Lpips_loss, Rec_loss и Reg_loss, которые используются для вычисления потерь.

Далее создается буфер шума и вычисляется среднее значение вектора w и его стандартное отклонение. Вектор w оптимизируется с использованием метода Adam. В цикле вычисляется синтезированное изображение, потери и производится обратное распространение ошибки. Результатом работы метода является оптимизированный вектор w.

В методе project_image_e4e используется предобученная модель e4e для получения начального латентого вектора изображения. Затем создаются экземпляры классов Lpips_loss, Rec_loss и Reg_loss, которые используются для вычисления потерь.

Далее создается буфер шума и оптимизируется латентый вектор изображения с использованием метода Adam. В цикле вычисляется синтезированное изображение, потери и производится обратное распространение ошибки. Результатом работы метода являются тензоры целевого изображения и оптимизированного латентого вектора.

### 2. Style transfer

Далее был использован метод для трансфера стиля с одного лица на другой с помощью смешивания векторов.

Были выбраны 3 стиля: аниме, классическое искусство и поп арт:

![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/68455776-0872-49b5-9252-638bdb4ab02e)

*Pic 1. Cтиль 1*

![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/b299fe6f-eaa4-42b8-b547-bc8910f46e7f)

*Pic 2. Стиль 2*

![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/9b6ab492-59bc-4243-b9ea-95fa15225c67)

*Pic 3. Стиль 3*

Стили так же были спроецированны в латентное пространство.


![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/79fc504a-a89a-4c98-9b08-e82fadf7c33d)
![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/d0a6394a-b046-4da1-b5d9-4632290e2ad2)
![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/21771115-bb45-4fd5-acc8-01a7418c4a69)

*Таблица 2. Style transfer*


В коде для трансфера стиля используется функция styles_crossover, которая принимает два латентых вектора latent1 и latent2, а также список индексов swap_dim_idxs. Функция заменяет значения в векторе latent1 на соответствующие значения из вектора latent2 по индексам, указанным в swap_dim_idxs.

В коде используются два диапазона индексов для замены значений в векторе latent1 на значения из вектора style_latent: list(range(10, 18)) и list(range(9, 18)).

Индексы в векторе W+ соответствуют различным уровням детализации изображения, от грубых черт до мелких деталей. Индексы с меньшими значениями соответствуют более грубым чертам, а индексы с большими значениями - более мелким деталям.

В данном случае, используя диапазон индексов list(range(10, 18)), мы заменяем значения в векторе latent1, отвечающие за более мелкие детали изображения, на соответствующие значения из вектора style_latent. Это позволяет перенести стиль изображения на более мелкие детали лица.

Аналогично, используя диапазон индексов list(range(9, 18)), мы заменяем значения в векторе latent1, отвечающие за более крупные детали изображения, на соответствующие значения из вектора style_latent. Это позволяет перенести стиль изображения на более крупные детали лица.

### 3. Expression Transfer

К сожалению не успел сделать(( 

*Таблица 3. Expression Transfer*



### 4. Face swap

Был использован уже написанный в ноутбуке Arcface лосс, скачаны веса модели по ссылке.

Он был добавлен в пайплайн оптимизации и с помощью оптимизации градиента айдентети с одной фотографии была перенесена на другую сохраняя при этом угол поворота и цвет для исходного лица.

![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/76c3d3b7-31cd-4676-9d9f-20e7d3c889c1)


*Таблица 4. Face swap*

(Визуализировал пересадку не всех со всеми, потому что получалось довольно громоздко.

