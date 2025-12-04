from abc import ABCMeta, abstractmethod
import re
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
    

nltk.download('stopwords')


class SteamDatasetVectorizer(metaclass=ABCMeta):
    @abstractmethod
    def vectorize_dataset(self, dataset: pd.DataFrame) -> np.ndarray[np.ndarray]:
        """
        Отображает каждую игру из датасета в численный вектор
        """
        pass 


class TfidfSteamDatasetVectorizer(SteamDatasetVectorizer):
    def __init__(self, max_features: int = 500, min_df: int = 2, max_df: float = 0.95):
        super().__init__()
        self.max_features = max_features
        self.min_df = min_df 
        self.max_df = max_df
        self.stop_words = set(stopwords.words("english"))

    
    def _process_short_description(self, text: str) -> str:
        """Предобработка описания игры"""
    
        if type(text) is not str: # обработка nan
            return ''
        
        text = text.lower()
        text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]', '', text)
        tokens = WordPunctTokenizer().tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return " ".join(tokens)

    
    def _process_categories(self, text: str) -> str:
        if type(text) is not str: # обработка nan
            return ''
        
        text = text.lower()
        # удаляем запятые
        text = [category.strip() for category in text.split(',')]
        return ' '.join(text)

    
    def _process_genres(self, text: str) -> str:
        if type(text) is not str: # обработка nan
            return ''
        
        text = text.lower()
        # удаляем запятые
        text = [category.strip() for category in text.split(',')]
        return ' '.join(text)

    
    def _process_tags(self, text: str) -> str:
        if type(text) is not str: # обработка nan
            return ''
    
        text = text.lower()
        text = [tag.strip().split(':')[0] for tag in text.split(',')]
        return ' '.join(text)

    
    def _merge(self, row):
        short_description = row['short_description'].split()
        categories = row['categories'].split()
        genres = row['genres'].split()
        tags = row['tags'].split()
    
        merged_genres_tags = list(set(genres) | set(tags))
    
        return ' '.join(short_description + categories + merged_genres_tags)

    
    def _preprocessing_pipeline(self, dataset: pd.DataFrame) -> pd.Series:
        data = dataset.copy()
        columns_to_save = ["app_id", "name", "short_description",
                           "categories", "genres", "tags"]

        data = data[columns_to_save]

        data['short_description'] = data['short_description'].apply(self._process_short_description)
        data['categories'] = data['categories'].apply(self._process_categories)
        data['genres'] = data['genres'].apply(self._process_genres)
        data['tags'] = data['tags'].apply(self._process_tags)
        
        corpus = data.apply(self._merge, axis=1)
        
        return corpus

    
    def vectorize_dataset(self, dataset: pd.DataFrame) -> np.ndarray[np.ndarray]:
        corpus = self._preprocessing_pipeline(dataset)

        vectorizer = TfidfVectorizer(max_features=self.max_features,
                                     min_df=self.min_df,     
                                     max_df=self.max_df)
        
        vectors = vectorizer.fit_transform(corpus).toarray()

        return vectors