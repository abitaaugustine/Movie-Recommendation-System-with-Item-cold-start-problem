import pandas as pd
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import logging # Use logging instead of print for better tracking

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "movies_mrs_multi_vector" # New collection name
DATA_PATH = os.path.join("data", "tmdb_movies_data.csv")
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DIMENSION = 384 # Dimension of the 'all-MiniLM-L6-v2' model
# Fields to embed separately
EMBED_FIELDS = ['original_title', 'overview', 'cast', 'genres']
SEARCH_FIELD = "overview_embedding" # Field to search against primarily

# --- Milvus Connection ---
def connect_to_milvus():
    """Establishes connection to Milvus."""
    try:
        logging.info(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        logging.info("Successfully connected to Milvus.")
        return True
    except Exception as e:
        logging.error(f"Failed to connect to Milvus: {e}")
        return False

def get_milvus_collection():
    """Gets the Milvus collection object, creating it if it doesn't exist."""
    if utility.has_collection(COLLECTION_NAME):
        logging.info(f"Collection '{COLLECTION_NAME}' found.")
        collection = Collection(name=COLLECTION_NAME)
        collection.load()
        return collection

    logging.info(f"Collection '{COLLECTION_NAME}' not found. Creating...")
    # Define fields: ID, text fields, and multiple embedding fields
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="original_title", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="overview", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="cast", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="director", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="tagline", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="genres", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="release_date", dtype=DataType.VARCHAR, max_length=20),
        # Separate embedding fields
        FieldSchema(name="title_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="overview_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="cast_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="genres_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
    ]
    schema = CollectionSchema(fields, description="Movie recommendation collection with multiple vectors")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    logging.info(f"Collection '{COLLECTION_NAME}' created.")

    # Create index for the primary search embedding field
    logging.info(f"Creating index on '{SEARCH_FIELD}'...")
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name=SEARCH_FIELD, index_params=index_params)
    # Optionally create indexes on other vector fields if needed for different search types
    # collection.create_index(field_name="title_embedding", index_params=index_params)
    # collection.create_index(field_name="cast_embedding", index_params=index_params)
    # collection.create_index(field_name="genres_embedding", index_params=index_params)
    logging.info("Index created.")
    collection.load()
    return collection

# --- Data Loading and Preprocessing ---
def load_and_prepare_data(filepath):
    """Loads movie data and prepares it."""
    try:
        abs_filepath = os.path.abspath(filepath)
        logging.info(f"Attempting to load data from: {abs_filepath}")
        df = pd.read_csv(abs_filepath)
        # Select relevant columns and handle missing values
        required_cols = ['original_title', 'overview', 'cast', 'director', 'tagline', 'genres', 'release_date']
        df = df[required_cols].fillna('')
        # Ensure text fields are strings
        for col in EMBED_FIELDS:
             df[col] = df[col].astype(str)
        logging.info(f"Loaded {len(df)} records.")
        return df
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {abs_filepath}")
        return None
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

# --- Embedding Generation ---
model = None
def get_embedding_model():
    """Loads the sentence transformer model."""
    global model
    if model is None:
        try:
            logging.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            model = SentenceTransformer(EMBEDDING_MODEL)
            logging.info("Embedding model loaded.")
        except Exception as e:
            logging.error(f"Error loading embedding model: {e}")
            return None
    return model

def generate_embeddings(texts):
    """Generates embeddings for a list of texts. Handles empty strings."""
    embedding_model = get_embedding_model()
    if not embedding_model:
        return None
    try:
        # Replace empty strings with a placeholder or handle them post-generation
        processed_texts = [text if text.strip() else " " for text in texts] # Use space for empty text
        embeddings = embedding_model.encode(processed_texts, show_progress_bar=True)
        return embeddings.tolist() # Convert to list for Milvus
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        return None

def generate_multiple_embeddings(df, fields_to_embed):
    """Generates separate embeddings for specified fields in a DataFrame."""
    embeddings_dict = {}
    total_records = len(df)
    for field in fields_to_embed:
        logging.info(f"Generating embeddings for field: {field} ({total_records} records)")
        texts = df[field].tolist()
        embeddings = generate_embeddings(texts)
        if embeddings is None:
            logging.error(f"Failed to generate embeddings for field {field}")
            return None # Abort if any embedding generation fails
        embeddings_dict[f"{field}_embedding"] = embeddings
        logging.info(f"Finished embeddings for field: {field}")
    return embeddings_dict

# --- Milvus Operations ---
def insert_data_to_milvus(collection, df, embeddings_dict):
    """Inserts data with multiple embeddings into the Milvus collection."""
    if df is None or embeddings_dict is None:
        logging.error("Invalid data or embeddings for insertion.")
        return

    # Prepare data for Milvus insertion
    data_to_insert = [
        df['original_title'].tolist(),
        df['overview'].tolist(),
        df['cast'].tolist(),
        df['director'].tolist(),
        df['tagline'].tolist(),
        df['genres'].tolist(),
        df['release_date'].astype(str).tolist(),
    ]
    # Add the embedding lists in the correct order based on schema
    data_to_insert.append(embeddings_dict['title_embedding'])
    data_to_insert.append(embeddings_dict['overview_embedding'])
    data_to_insert.append(embeddings_dict['cast_embedding'])
    data_to_insert.append(embeddings_dict['genres_embedding'])

    try:
        logging.info(f"Inserting {len(df)} records into Milvus...")
        mr = collection.insert(data_to_insert)
        collection.flush() # Ensure data is written
        logging.info(f"Successfully inserted data. Primary keys count: {len(mr.primary_keys)}")
        logging.info(f"Total entities in collection: {collection.num_entities}")
    except Exception as e:
        logging.error(f"Error inserting data into Milvus: {e}")

def search_similar_movies(collection, query_embedding, search_field=SEARCH_FIELD, top_k=5):
    """Searches for similar movies based on a query embedding against a specific field."""
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    try:
        logging.info(f"Searching top {top_k} similar movies using field '{search_field}'...")
        results = collection.search(
            data=[query_embedding], # Search query embedding
            anns_field=search_field, # The vector field to search in
            param=search_params,
            limit=top_k,
            output_fields=["original_title", "overview", "cast", "director", "genres", "release_date"] # Text fields to return
        )
        logging.info("Search completed.")
        # Process results
        hits = results[0]
        search_results = []
        for hit in hits:
            # Adjust based on your Milvus version if entity structure differs
            entity_data = hit.entity.to_dict()['entity']
            search_results.append({
                "id": hit.id,
                "distance": hit.distance,
                "title": entity_data.get('original_title', 'N/A'),
                "overview": entity_data.get('overview', 'N/A'),
                "cast": entity_data.get('cast', 'N/A'),
                "director": entity_data.get('director', 'N/A'),
                "genres": entity_data.get('genres', 'N/A'),
                "release_date": entity_data.get('release_date', 'N/A')
            })
        return search_results
    except Exception as e:
        logging.error(f"Error searching in Milvus: {e}")
        return []

def add_new_movie(collection, movie_data):
    """Adds a single new movie with multiple embeddings to Milvus."""
    # Generate embeddings for each relevant field
    embeddings_dict = {}
    primary_embedding = None # Store the embedding used for immediate search
    for field in EMBED_FIELDS:
        text = movie_data.get(field, "")
        embedding = generate_embeddings([text]) # Generate embedding for single text
        if embedding is None:
            logging.error(f"Failed to generate embedding for field {field} for the new movie.")
            return None, None # Indicate failure
        embeddings_dict[f"{field}_embedding"] = embedding
        if f"{field}_embedding" == SEARCH_FIELD:
            primary_embedding = embedding[0] # Get the vector itself

    if not primary_embedding:
         logging.error(f"Could not get primary search embedding ({SEARCH_FIELD}) for the new movie.")
         return None, None

    # Prepare data for insertion (list of lists for each field)
    data_to_insert = [
        [movie_data['original_title']],
        [movie_data['overview']],
        [movie_data['cast']],
        [movie_data['director']],
        [movie_data['tagline']],
        [movie_data['genres']],
        [str(movie_data['release_date'])],
        embeddings_dict['title_embedding'],
        embeddings_dict['overview_embedding'],
        embeddings_dict['cast_embedding'],
        embeddings_dict['genres_embedding'],
    ]

    try:
        logging.info("Inserting new movie...")
        mr = collection.insert(data_to_insert)
        collection.flush()
        new_id = mr.primary_keys[0]
        logging.info(f"Successfully inserted new movie with ID: {new_id}")
        # Return the ID and the primary embedding for immediate search
        return new_id, primary_embedding
    except Exception as e:
        logging.error(f"Error inserting new movie: {e}")
        return None, None


# --- Initial Setup ---
def initial_setup():
    """Connects to Milvus, creates collection, and loads initial data if needed."""
    if not connect_to_milvus():
        return None, None

    collection = get_milvus_collection()

    if collection.num_entities == 0:
        logging.info("Collection is empty. Performing initial data load...")
        df = load_and_prepare_data(DATA_PATH)
        if df is not None:
            # Load a subset initially
            df_subset = df.head(1000)
            logging.info(f"Generating multiple embeddings for {len(df_subset)} movies...")
            embeddings_dict = generate_multiple_embeddings(df_subset, EMBED_FIELDS)
            if embeddings_dict:
                insert_data_to_milvus(collection, df_subset, embeddings_dict)
            else:
                logging.error("Failed to generate embeddings for initial data.")
        else:
             logging.warning("Failed to load initial data. Collection remains empty.")
    else:
        logging.info(f"Collection '{COLLECTION_NAME}' already contains {collection.num_entities} entities.")

    return collection, get_embedding_model()


if __name__ == "__main__":
    logging.info("Running main.py directly (for testing initial setup)...")
    collection, model = initial_setup()
    if collection and model:
        logging.info("Initial setup successful.")
    else:
        logging.error("Initial setup failed.")
    # connections.disconnect("default") # Disconnect if running as a standalone script
    # logging.info("Disconnected from Milvus.")