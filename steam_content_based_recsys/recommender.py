import os
import json
from dataclasses import dataclass 
import numpy as np
import pandas as pd
import faiss
from .exceptions import *
from .vectorizers import SteamDatasetVectorizer


@dataclass
class ContentBasedRecommendation:
    original: int # исходная игра
    recommendation: int # похожая на нее
    similarity: float # коэффициент похожести


class SteamContentBasedRecSys:
    def __init__(self):
        self.required_columns = [
            'app_id', 'name', 'release_date', 'required_age', 
            'price', 'short_description', 'windows', 'mac', 
            'linux', 'developers', 'publishers', 'categories', 
            'genres', 'supported_languages', 'tags', 'min_package_price'
        ]
        self.fitted = False

    def _validate_dataset(self, dataset: pd.DataFrame) -> bool:
        """
        Если датасет содержит все необходимые столбцы -> True
        Иначе бросается исключение InvalidSteamDataset
        """
        for col in self.required_columns:
            if col not in dataset:
                raise InvalidSteamDataset(f"Missing column: {col}")
        return True
        
    def fit(self, vectorizer: type, dataset: pd.DataFrame):
        """Обучение рекомендательной системы"""
        # валидация входных данных
        self._validate_dataset(dataset)
        if not issubclass(vectorizer, SteamDatasetVectorizer):
            raise InvalidVectorizerType("vectorizer must be subclass of SteamDatasetVectorizer class")
        
        # Сопоставления между id игр и их индексами в таблице (пригодится при сравнении векторов)
        self.idx_to_id = dataset['app_id'].to_dict()
        self.id_to_idx = {item[1]: item[0] for item in self.idx_to_id.items()}

        # получаем векторы
        vectors = vectorizer().vectorize_dataset(dataset)
        
        assert vectors.shape[0] == dataset.shape[0], "Количество игр должно совпадать!"
        
        self.index = faiss.IndexFlatIP(vectors.shape[1])  # индекс для расчета косинусной близости
        self.index.add(vectors)
        self.fitted = True
        return self

    def save(self, index_path: str = './steam.index', idx2id_path: str = 'idx2id.json') -> None:
        """
        * index_path: str - путь до .index файла, куда будет сохранена векторная база данных
        * idx2id_path: str - путь до .json файла, куда будет сохранено соответствие между id игр и их индексами в БД
        """
        if not index_path.endswith('.index'):
            raise InvalidPath(f"Index file must have a .index format")
            
        if not self.fitted:
            raise NotFitted("Fit the system first via .fit() method")
            
        if not idx2id_path.endswith('.json'):
            raise InvalidPath(f"idx2id path must have a .json format")
            
        faiss.write_index(self.index, index_path)
        with open(idx2id_path, 'w') as idx2id_file:
            json.dump(self.idx_to_id, idx2id_file)

    def load(self, index_path: str, idx2id_path: str):
        if not index_path.endswith('.index'):
            raise InvalidPath(f"Index file must have a .index format")
            
        if not idx2id_path.endswith('.json'):
            raise InvalidPath(f"idx2id path must have a .json format")

        if not os.path.exists(index_path):
            raise InvalidPath(f"File: {index_path} does not exists")

        if not os.path.exists(idx2id_path):
            raise InvalidPath(f"File: {idx2id_path} does not exists")

        self.index = faiss.read_index(index_path)
        with open(idx2id_path, 'r') as idx2id_file:
            self.idx_to_id: dict = json.load(idx2id_file)
        self.id_to_idx = {item[1]: item[0] for item in self.idx_to_id.items()}
        self.fitted = True
        return self

    
    def _get_game_vector(self, game_id: int) -> np.ndarray:
        if game_id not in self.id_to_idx:
            raise GameNotFound(f"Game with id={game_id} does not exists")
        game_idx = self.id_to_idx[game_id]
        return self.index.reconstruct(game_idx)

    
    def _find_similar_games(self, games: list[int], k: int = 10) -> tuple[np.ndarray[np.ndarray], np.ndarray[np.ndarray]]:
        """Поиск игр похожих на заданные"""
        vectors = []
        for game in games:
            vectors.append(self._get_game_vector(game))
        vectors = np.array(vectors)
    
        distances, indexes = self.index.search(vectors, k+1)  # +1 чтобы исключить исходные игры
        return distances[:, 1:], indexes[:, 1:]


    def recommend(self, liked: list[int], M: int) -> list[ContentBasedRecommendation]:
        """Формирование content-based рекомендации"""
        distances, indexes = self._find_similar_games(liked, M)
        merged = []
        for i in range(len(liked)):
            liked_i = liked[i]
            rec_i = indexes[i]
            dists_i = distances[i]
            for j in range(M):
                merged.append((liked_i, rec_i[j], dists_i[j]))
                
        merged = sorted(merged, key=lambda x: x[2], reverse=True)
    
        answer = []
        liked = set(liked)
        for x in merged:
            if x[1] not in liked:
                answer.append(ContentBasedRecommendation(original=x[0], recommendation=x[1], similarity=x[2]))
                if len(answer) == M:
                    break
        return answer