# Steam Recommendation System
Это модуль content-based рекомендательной системы. 

Модуль можно установить через pip:
```bash
pip install git+https://github.com/yourusername/steam-recsys.git
```

Ну или просто взять папочку `steam_content_based_recsys/` и скопировать куда нужно :) (Не забыв установить зависимости из requirements.txt само собой)

## API модуля
```python
import pandas as pd
from steam_content_based_recsys import SteamContentBasedRecSys, TfidfSteamDatasetVectorizer

# читаем наш исходный датасет
data = pd.read_csv("../combined_steam_games.csv")

# обучаем рекомендательную систему
recsys = SteamContentBasedRecSys().fit(TfidfSteamDatasetVectorizer, data)


# имитация списка игр, которые понравились пользователю
liked_games = [20200, 655370, 1732930, 1355720, 1139950]

# получаем 5 рекомендаций, похожих на игры из списка выше
recommendations = recsys.recommend(liked_games, M=5)

for sample in recommendations:
    print(f"Игра с id: {sample.recommendation} похожа на игру {sample.original} с коэффициентом = {sample.similarity}")
```


`SteamContentBasedRecSys.fit()` принимает на вход два аргумента:
- тип используемого векторизатора (пока он в модуле только один - TfidfSteamDatasetVectorizer) 
- датасет 

После обучения можно воспользоваться методом `.save()`, чтобы сохранить полученную векторную БД, а также соответствие между id игр и их индексами в этой БД:
```python
recsys.save('game.index', 'idx2id.json')
```

А потом вместо повторного обучения можно инициализировать рекомендательную систему из этих файлов:
```python
recsys.load('game.index', 'idx2id.json')
```
