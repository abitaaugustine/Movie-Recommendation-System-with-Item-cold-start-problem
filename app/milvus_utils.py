from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
import logging


class MilvusManager:
    def __init__(self, host="localhost", port="19530", collection_name="movies", dim=384):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self.collection = None

    def connect(self):
        try:
            connections.connect("default", host=self.host, port=self.port)
            logging.info("Connected to Milvus")
        except Exception as e:
            logging.error(f"Failed to connect to Milvus: {e}")
            raise

    def create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="overview", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="tagline", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="genres", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=500),
             FieldSchema(name="cast", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="crew", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="release_date", dtype=DataType.VARCHAR, max_length=100),
            *[
                FieldSchema(name=f"{field}_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
                for field in ["title", "overview", "tagline", "genres", "keywords", "cast", "crew"]
            ]
        ]
        schema = CollectionSchema(fields)
        self.collection = Collection(name=self.collection_name, schema=schema)
        logging.info(f"Collection `{self.collection_name}` created.")
        return self.collection

    def get_collection(self):
        if self.collection_name in Collection.list_collections():
            self.collection = Collection(self.collection_name)
        else:
            self.collection = self.create_collection()
        return self.collection

    def initialize(self):
        self.connect()
        return self.get_collection()

    def insert_movies(self, data_dict):
        self.collection = self.collection or self.get_collection()
        entities = [data_dict[key] for key in [
            "id", "original_title", "overview", "tagline", "genres", "keywords",
            "cast", "crew", "release_date",
            "title_vector", "overview_vector", "tagline_vector", "genres_vector",
            "keywords_vector", "cast_vector", "crew_vector"
        ]]
        self.collection.insert(entities)
        logging.info(f"Inserted {len(data_dict['id'])} movies into Milvus")

    def search_similar(self, vector, field="title_vector", top_k=5, filter_criteria=None):
        self.collection.load()
        expr = f"{filter_criteria[0]} == '{filter_criteria[1]}'" if filter_criteria else None
        results = self.collection.search(
            data=[vector],
            anns_field=field,
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            expr=expr,
            output_fields=["id", "title", "overview", "tagline", "genres", "keywords", "cast", "crew", "release_date"]
        )
        return results[0]
