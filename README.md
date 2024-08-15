# RL Algorithm Distillation
Моя попытка имплементации и в целом видение метода дистилляции RL алгоритмов, представленного в [In-context Reinforcement Learning with Algorithm Distillation](https://arxiv.org/abs/2210.14215) + мини обзор статей (все для [dunnolab](http://t.me/dunnolab))

## Среда
В качестве среды была выбрана простая среда для meta-RL - [DarkRoom](https://github.com/corl-team/toy-meta-gym/blob/main/src/toymeta/dark_room.py). 
Параметры среды (старался сделать приближенными к оригинальной статье по AD):
- `size = 9` - среда содержит 80 тасок (без награды по центру, когда агент автоматически выйграл)
- `random_start = False` - агент начинает всегда с центра карты
- `terminate_on_goal = True` - когда агент находит выход, среда перезагружается

Также для удобства обучения и работы со средой написал обертку для ограничения длины эпизода (значение в `info`, а также функционал `truncate`)
## RL алгоритм
Для сбора историй обучений был выбрал A3C (aka source algorithm). Тот же DQN и его производные не захотел выбирать, так как он на моем опыте сильно зависит от инициализации и порой неустойчиво учится (+ лично люблю более продвинутые PG методы). 
Архитектура: первый эмбед слой - общий для актора и критика, далее по линейному слою.

Особенность имплементации:
- актор не копируется в окружение, где запущена среда; среды живут в своих окружениях и в них отправляются действия, оттуда обратно возвращаются степы (если не ошибаюсь, в оригинале A3C вся сетка актора целиком отправляется в окружение, там собирает роллаут, делается синк градиентов на основном процессе, а потом заново все перекопируется - что для маленьких сред и агентов очевидно не имеет смысла) - мб валиднее все же говорить, что это A2C...
- Часто используют GAE оценки, я использовал оценку максимальной длины для value (разницы быть не должно, так как в A3C не используются длинные роллауты)

Параметры (старался сделать приближенными к оригинальной статье по AD):
- `hidden_size = 128` - размер эмбед-слоя
- `nenvs = 10` - запускается 10 параллельных сред
- `nsteps = 10` - длина роллаута
- `env_steps = 2_500_000` - всего со сред собирается 2,5М степов, что заведомо больше, чем необходимо алгоритму до сходимости к оптимальному (при обучении GPT - истории обрезаются)
- `optimizer = Adam`:
    - `lr = 1e-4`
    - `betas = (0.9, 0.999)`
    - `eps = 1e-6`
- `clip_grad_norm = 0.5`
- лось - комбинированный для актора и для критика (сетка же одна):
    - `entropy_coef = 0.01` - энтропийная регуляризация (aka advanced exploration)
    - `value_coef = 0.99` - коэффициент на лосс критика

Было обучено 57 агентов (по одному на таску), остальные таски - тестовые. В историях обучений также сохранялся timestep для использования в GPT (см. ниже). Для каждого агента после обучения сохраняется видео его хождения в среде (через ffmpeg).

## In-Context RL
В качестве трансформера взял [Decision Transformer](https://github.com/corl-team/CORL/blob/main/algorithms/offline/dt.py).

Особенности: 
- используется не такое абсолютное позиционное кодирование, как в оригинале - в качестве time_step подается time прямо из среды (чтобы наверняка передавать агенту инфу, что скоро произойдет перезагрузка среды). Оригинальная идея хранить для каждого time_step свой эмбед вектор, когда time_step длинные (истории обучения же довольно длинные), мне показалось странной затеей, так как, может произойти так, что в историях обучения при больших тайм-степах действия *более оптимальные* (что скорее всего так и есть), тогда это повлияет и на инференс - непонятно какие тайм-степы лучше подавать (хотя безусловно было бы неплохо это затестить влияет ли это вообще) - получилось что-то типо relative positional encoding для RL сред с ограниченными по длине эпизодами.
- reward-to-go/sum_rewards не используется... Просто подаю reward, который пришел из среды. Также так как в среде только 2 варианта наград - 0 или 1, то они идут в эмбед-слой (это чуть больше параметров чем в оригинале, но умножение вектора на 0 никуда не годится, тем более, что оно потом в атеншн идет, а делать сдвиг на reward мне показалось идеей похуже)

Параметры (старался сделать приближенными к оригинальной статье по AD):
- `history_len = 800_000` - истории обучения обрезаются, как было сказано выше, после 800к агенты уже особо ничему не учатся и быстро находят оптимальный путь до награды.
- `batch_size = 32`
- `seq_len = 256` - из историй вытягиваются 256 последовательных переходов (всего 256*3 токенов)
- `embedding_dim = 64`
- `num_layers = 4` - 4 атеншн слоя
- `feed_forward_dim = 32 * embedding_dim = 2048` - размерность скрытого слоя в feed forward
- `num_heads = 4`
- `attention_dropout = 0.5`
- `residual_dropout = 0.1`
- `embedding_dropout = 0`
- `optimizer = Adam`:
    - `lr = 3e-4`
    - `betas = (0.9, 0.99)`
- `clip_grad_norm = 1`
- `scheduler = Cosine Decay`:
    - `min_lr = 2e-6`
    - `num_steps = 22k` - что соответсвует в среднем посмотреть на каждый степ в историях по 4 раза
- `eval_step = 64` - каждые 64 итерации обучения гпт в ин-контекст режиме отправляется гулять по среде
- `eval_len = 256` - и гуляет она там 256 степов

# Код
При обучении A3C для сред помимо обертки для паралелльного взаимодействия со средами также написана обертка для логирования статистик наград и длины эпизода и тд в tensorboard. Процесс обучения каждого агента также логируется в tensorboard. Все логи по итогу будут в папке `src/logs/`.

После обучения в папке `src/videos/` создаются итоговые видео оптимальных агентов.

Истории обучения сохраняются в `src/histories/` в .txt формате

Тестовые таски в `src/test_tasks.txt`

- `search_params.ipynb` - ноутбук, в котором подбирал параметры для обучения A3C (на таске, когда дверь в левом верхнем углу `goal = [0, 0]`)
- `train_agents.ipynb` - ноутбук, из которого запускал обучение всех агентов
- `train_gpt.ipynb` - запуск обучения DT
- `src/utils/` - весь код лежит тут
- `src/utils/env_batch.py` - логика взаимодействия со средами в параллельном режиме
- `src/utils/wrappers.py` - обертки над средами + функция создания "параллельной мега-среды"
- `src/utils/runners.py` - вспомогательный класс для взаимодействия сетки с параллельными средами
- `src/utils/custom_transforms.py` - расчет оценки V-функции, а также мержинг роллаутов
- `src/utils/a2c.py` - сама архитектура A3C и град степ
- `src/utils/train_a2c.py` - вспомогательная функция для запуска обучения одного агента на определенную таску
- `dt.py` - архитектура DT
- `utils.py` - все остальные вспомагательные функции (eval для A3C и eval для DT)

Google-диск со всеми видео и tensorboard ивентами: [ссылка](https://drive.google.com/drive/folders/1E7glxXxkCXufiq6AubQPyqZ-wvX2Rg_-?usp=sharing)

- eval у A3C - усредненные значения по параллельным средам (награды также усреднены за последние 10 эпизодов в каждой среде)

- eval у DT - усредненные значения последних 10 эпизодов в одной среде на тестовую таску (той, которой нет в истории обучения офк) - см. ниже

# bloopers
Обучение DT в той конфигурации, в которой я описал занимает 24ч+ - я не успел доучить, так что результаты так себе((( 
    
Что также обидно - я накосячил с аргументами в tensorboard при обучении DT (и не успел исправить). По этой причине eval метрики при обучении DT превратились в кашу в борде 
+
не написал kv-cache((( - eval ограничен в возможностях и совсем неэффективен в данной имплементации
+
также не хватило времени для имплементации адекватного батчированного eval на разные тестовые таски с разных сред, поэтому случайность eval того, которого имеем, не исключена...


# Summary
Если говорить в целом про метод, то звучит он как мечта - и эффективность по количеству сэмплов в среде, и быстрая адаптация-генерализация под новые таски, но не лишен недостатков - непонятно как генерализоваться на новые стейты, если поменять размер среды с 9 там до 13, то все, embed слой не потянет (для новых действий способ кнч есть, это здорово, но со стейтами не особо понятно что делать)
Хотелось бы исследовать насколько быстрый темп обучения должен быть в историях и для разных по сложности сред. Плюс на данный момент в DT 3 модальности (r, s, a) конкатенируются в одну и далее подаются в атеншн (с маской но тем не менее) - мне кажется это странным и неоптимальным способом использования по природе разных модальностей, можно бы использовать какой-нибудь хитрый cross-attention между модальностями (inspired by AlphaFold2) - не уверен, что это на что-то повлияет, но для некоторых сред возможно сделает обучение трансформера быстрее и оптимальнее.
Также не очень нравится использование втупую энтропийного лосса на историях - быть может можно придумать что-то на основе теории нестационарных MDP?

# Разбор статей
1. [Transformers Learn Temporal Difference Methods
for In-Context Reinforcement Learning](https://arxiv.org/pdf/2405.13861)

Очень тяжелая статья, в которой авторы теоретически показывают, что трансформеры умеют в in-context RL (эмпирически тоже, но экспериментов было мало, упор в теорию). Прямо в in-context формате трансформеры способны воспроизводить разные "алгоритмы" - residual gradient, temporal difference и average reward TD learning. Все подтверждается теоремами с довольно большими и неприятными доказательствами, но это все для упрощенной модели Markov Reward Process (без действий). Статья довольно фундаментальна, потому что дает подверждение, что in-context RL может выполнять разные классы алгоритмов как бы *на лету*, а возможно может и большее (типо те возможные алгоритмы которые человечеству неизвестны, но они могут существовать).

2. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290)

Based. В моем понимании это довольно легендарная статья. Авторы из OpenAI представили метод RLHF (reward modelling на простом предположении + PPO), всем понравилось и пайплайн и метод. Но вот по итогу PPO по-моему особо ни у кого кроме OpenAI и не запустился при language modelling. В этой статьей авторы практически просто подставляют одно уравнение в другое и получают метод, который уже гораздо больше используется людьми в GPT-like models (вроде бы он чаще действительно делает alignment нежели оригинальный RLHF). Также дают какой-то небольшой анализ метода, и главное сравнение, в котором DPO превосходит PPO.

3. [UNLEASHING THE POWER OF PRE-TRAINED LANGUAGE MODELS FOR OFFLINE REINFORCEMENT LEARNING](https://arxiv.org/pdf/2310.20587)

Очень неожиданная статья, в которой претрейн ллм, тренированные на википедии, после этого тренируют на RL средах, как DT с LoRA модулями... Звучит, как лютый кринж, и забавно, что это вообще аутперформит... аутперформит даже оригинальный DT!! Самое удивительное: 1. просто вместо слов начали подавать то стейты, то реворды (о чем нужно думать, чтобы даже случайно сделать именно это??), 2. в лосс подмешивают обычный gpt-шный языковой лосс и вот именно с ним получается еще доп прирост в качестве.

В общем, эта статья ломает мои шаблоны того, как нужно делать МЛ и РЛ. Или это все продолжение 7 рукопожатий и small worlds????
