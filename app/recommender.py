from .milvus_utils import MilvusManager
from .embedding import TextEmbedder


class MovieRecommender:
    def __init__(self, collection_name="movies"):
        self.embedder = TextEmbedder()
        self.milvus = MilvusManager(collection_name=collection_name)
        self.collection = self.milvus.get_collection()

    def recommend_similar_movies(self, movie_data: dict, top_k=5):
        combined_results = {}
        for field in ["title", "overview", "tagline", "genres", "keywords", "cast", "crew"]:
            if movie_data.get(field):
                vector = self.embedder.embed_text(movie_data[field])
                results = self.milvus.search_similar(vector, field=f"{field}_vector", top_k=top_k)
                for result in results:
                    movie_id = result.id
                    score = result.distance
                    combined_results[movie_id] = combined_results.get(movie_id, 0) + (1 - score)

        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in sorted_results[:top_k]]
