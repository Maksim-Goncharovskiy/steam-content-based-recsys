from abc import ABCMeta, abstractmethod
import re
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from gensim.models.doc2vec import Doc2Vec
    

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



class Doc2VecSteamDatasetVectorizer(SteamDatasetVectorizer):
    def __init__(self, doc2vec_checkpoint: str):
        super().__init__()
        self.doc2vec = Doc2Vec.load(doc2vec_checkpoint)

    def _extract_year(self, date: str) -> int:
        if type(date) is not str:
            return 0
        if ',' in date:
            return int(date.split(',')[-1].strip())
        else:
            return int(date.split()[-1].strip())

    def _one_hot_encode(self, df, column_name):
        df_copy = df.copy()
        
        df_copy[column_name] = df_copy[column_name].str.replace(' ', '')
        
        split_categories = df_copy[column_name].str.split(',')
        
        exploded = split_categories.explode()
        
        dummies = pd.get_dummies(exploded, prefix=column_name).astype(float)
        
        result = dummies.groupby(level=0).max()
        
        final_df = pd.concat([df_copy, result], axis=1)
        
        return final_df
        
    def vectorize_dataset(self, dataset: pd.DataFrame) -> np.ndarray[np.ndarray]:
        data = dataset.copy()
        data = data[[
            'release_date', 
            'price', 
            'short_description', 
            'windows',
            'mac', 'linux', 'categories', 'genres']]
        
        # 1. предобработка года
        data['year'] = data['release_date'].apply(self._extract_year)
        data = data.drop(['release_date'], axis=1)

        # 2. предобработка цены
        data['is_free'] = data['price'].apply(lambda x: 1 if x == 0.0 else 0)
        data['<10'] = data['price'].apply(lambda x: 1 if x < 10 else 0)
        data['>=10'] = data['price'].apply(lambda x: 1 if x >= 10 else 0)
        data = data.drop(['price'], axis=1)
        
        # 3. предобработка категорий и жанров
        data = self._one_hot_encode(data, 'categories')
        data = self._one_hot_encode(data, 'genres')
        data = data.drop(['categories', 'genres'], axis=1)
        
        # 4. предобработка столбцов с ОС
        data['windows'] = data['windows'].astype(float)
        data['linux'] = data['linux'].astype(float)
        data['mac'] = data['mac'].astype(float)

        # 5. Понижение размерности на категориальных признаках
        data = data.drop(['short_description'], axis=1)
        cat = StandardScaler().fit_transform(data.values)
        pca = PCA(n_components=28)
        cat = pca.fit_transform(cat)
        
        # 5. эмбеддинги описаний
        text_embeddings = self.doc2vec.dv.vectors
        
        # 6. формирование итоговых векторных представлений
        vectors = np.concatenate([cat, text_embeddings], axis=1)
        
        return vectors