from sentence_transformers import SentenceTransformer
import numpy as np


class TextEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text) if text else np.zeros(self.dim)

    def embed_keywords(self, keywords: str) -> np.ndarray:
        return self.embed_text(" ".join(keywords.split('|')))

    def embed_movie_data(self, row: dict) -> dict:
        data = {
            'id': int(row['id']),
            'original_title': row.get('original_title', ''),
            'cast': row.get('cast', ''),
            'director': row.get('director', ''),
            'tagline': row.get('tagline', ''),
            'keywords': row.get('keywords', ''),
            'overview': row.get('overview', ''),
            'genres': row.get('genres', ''),
            'release_year': row.get('release_year', 0),
        }

        data['title_vector'] = self.embed_text(data['original_title'])
        data['overview_vector'] = self.embed_text(data['overview'])
        data['tagline_vector'] = self.embed_text(data['tagline'])
        data['keywords_vector'] = self.embed_keywords(data['keywords'])
        data['cast_vector'] = self.embed_keywords(data['cast'])
        data['genres_vector'] = self.embed_keywords(data['genres'])

        return data
