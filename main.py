import pandas as pd
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import logging # Use logging instead of print for better tracking
from collections import defaultdict
import math # For rank aggregation scoring

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

# --- In-Memory User Profile Store ---
# Note: This is non-persistent. For production, use a database.
user_profiles = {} # { user_id: {"name": "...", "preferences": set([movie_id1, movie_id2, ...])} }


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
        # Ensure collection is loaded before use
        if not collection.has_index(index_name=f"_{SEARCH_FIELD}_index"): # Check if index exists (adjust name if needed)
            logging.warning(f"Index not found on {SEARCH_FIELD}. Attempting to create.")
            # Re-create index if missing (consider if this is desired behavior)
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            try:
                collection.create_index(field_name=SEARCH_FIELD, index_params=index_params)
                logging.info(f"Index created on {SEARCH_FIELD}.")
            except Exception as e:
                logging.error(f"Failed to create index: {e}")
                # Decide how to handle this - maybe return None or raise error
        try:
            collection.load()
            logging.info(f"Collection '{COLLECTION_NAME}' loaded.")
        except Exception as e:
            logging.error(f"Failed to load collection '{COLLECTION_NAME}': {e}")
            return None # Cannot proceed without loaded collection
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
    try:
        collection.create_index(field_name=SEARCH_FIELD, index_params=index_params, index_name=f"_{SEARCH_FIELD}_index") # Naming the index
        # Optionally create indexes on other vector fields if needed for different search types
        collection.create_index(field_name="title_embedding", index_params=index_params)
        collection.create_index(field_name="cast_embedding", index_params=index_params)
        collection.create_index(field_name="genres_embedding", index_params=index_params)
        logging.info("Index created.")
        collection.load()
        logging.info(f"Collection '{COLLECTION_NAME}' loaded after creation.")
    except Exception as e:
        logging.error(f"Failed to create index or load collection after creation: {e}")
        return None
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
        for col in EMBED_FIELDS + ['director', 'tagline', 'release_date']: # Include non-embedded text fields
            if col in df.columns:
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
        processed_texts = [text if text and text.strip() else " " for text in texts] # Use space for empty text
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

    # Prepare data for Milvus insertion - ensure order matches schema
    data_to_insert = [
        df['original_title'].tolist(),
        df['overview'].tolist(),
        df['cast'].tolist(),
        df['director'].tolist(),
        df['tagline'].tolist(),
        df['genres'].tolist(),
        df['release_date'].astype(str).tolist(), # Ensure release_date is string
        # Add the embedding lists in the correct order based on schema
        embeddings_dict['title_embedding'],
        embeddings_dict['overview_embedding'],
        embeddings_dict['cast_embedding'],
        embeddings_dict['genres_embedding'],
    ]

    try:
        logging.info(f"Inserting {len(df)} records into Milvus...")
        mr = collection.insert(data_to_insert)
        collection.flush() # Ensure data is written
        logging.info(f"Successfully inserted data. Primary keys count: {len(mr.primary_keys)}")
        logging.info(f"Total entities in collection after insert: {collection.num_entities}")
        return mr.primary_keys # Return the IDs of inserted entities
    except Exception as e:
        logging.error(f"Error inserting data into Milvus: {e}")
        return None


def search_similar_movies(collection, query_embedding, search_field=SEARCH_FIELD, top_k=5, exclude_ids=None):
    """Searches for similar movies based on a query embedding against a specific field, optionally excluding IDs."""
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    # Build expression to exclude IDs if provided
    expr = None
    if exclude_ids:
        exclude_ids_list = list(exclude_ids) # Ensure it's a list
        if exclude_ids_list: # Only add filter if list is not empty
            expr = f"id not in {exclude_ids_list}"
            logging.info(f"Excluding IDs: {exclude_ids_list}")


    try:
        logging.info(f"Searching top {top_k} similar movies using field '{search_field}'...")
        results = collection.search(
            data=[query_embedding], # Search query embedding
            anns_field=search_field, # The vector field to search in
            param=search_params,
            limit=top_k + (len(exclude_ids) if exclude_ids else 0), # Fetch more initially to account for filtering
            expr=expr, # Apply the filter expression
            output_fields=["id", "original_title", "overview", "cast", "director", "genres", "release_date"] # Text fields to return
        )
        logging.info("Search completed.")
        # Process results
        hits = results[0]
        search_results = []
        count = 0
        for hit in hits:
            # Adjust based on your Milvus version if entity structure differs
            # For newer versions (like 2.x), entity might be directly accessible
            entity_data = hit.entity if hasattr(hit, 'entity') else hit # Adjust based on actual hit object structure
            # If entity_data is not a dict, try accessing fields directly
            movie_id = hit.id
            # Skip if this ID was meant to be excluded (double check, though expr should handle it)
            if exclude_ids and movie_id in exclude_ids:
                continue

            search_results.append({
                "id": movie_id,
                "distance": hit.distance,
                # Use .get() for safety if entity_data might not be a dict or fields might be missing
                "title": entity_data.get('original_title', 'N/A') if isinstance(entity_data, dict) else getattr(entity_data, 'original_title', 'N/A'),
                "overview": entity_data.get('overview', 'N/A') if isinstance(entity_data, dict) else getattr(entity_data, 'overview', 'N/A'),
                "cast": entity_data.get('cast', 'N/A') if isinstance(entity_data, dict) else getattr(entity_data, 'cast', 'N/A'),
                "director": entity_data.get('director', 'N/A') if isinstance(entity_data, dict) else getattr(entity_data, 'director', 'N/A'),
                "genres": entity_data.get('genres', 'N/A') if isinstance(entity_data, dict) else getattr(entity_data, 'genres', 'N/A'),
                "release_date": entity_data.get('release_date', 'N/A') if isinstance(entity_data, dict) else getattr(entity_data, 'release_date', 'N/A')
            })
            count += 1
            if count >= top_k:
                break # Stop once we have top_k results after filtering

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

    # Prepare data for insertion (list of lists for each field) - ensure order matches schema
    data_to_insert = [
        [movie_data.get('original_title', '')],
        [movie_data.get('overview', '')],
        [movie_data.get('cast', '')],
        [movie_data.get('director', '')],
        [movie_data.get('tagline', '')],
        [movie_data.get('genres', '')],
        [str(movie_data.get('release_date', ''))], # Ensure string
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


# --- User Profile Functions ---
def get_or_create_user(user_id, name=""):
    """Gets or creates a user profile in the in-memory store."""
    if user_id not in user_profiles:
        user_profiles[user_id] = {"name": name if name else f"User_{user_id}", "preferences": set()}
        logging.info(f"Created profile for user ID: {user_id}")
    return user_profiles[user_id]


def add_movie_preference(user_id, movie_id):
    """Adds a movie ID to the user's preferences."""
    profile = get_or_create_user(user_id) # Ensure user exists
    # Optional: Check if movie_id exists in Milvus first (requires a query)
    # result = collection.query(expr=f"id == {movie_id}", limit=1)
    # if not result:
    #     logging.warning(f"Attempted to add non-existent movie ID {movie_id} to preferences for user {user_id}.")
    #     return False

    if movie_id not in profile["preferences"]:
        profile["preferences"].add(movie_id)
        logging.info(f"Added movie ID {movie_id} to preferences for user {user_id}.")
        return True
    else:
        logging.info(f"Movie ID {movie_id} already in preferences for user {user_id}.")
        return False


def get_movie_embeddings_by_ids(collection, movie_ids, embedding_field=SEARCH_FIELD):
    """Retrieves embeddings for a list of movie IDs from Milvus."""
    if not movie_ids:
        return []
    # Ensure IDs are integers if your schema expects INT64
    int_movie_ids = [int(mid) for mid in movie_ids]
    expr = f"id in {list(int_movie_ids)}"
    try:
        results = collection.query(
            expr=expr,
            output_fields=["id", embedding_field]
        )
        # Create a dictionary mapping ID to embedding
        embedding_map = {res['id']: res[embedding_field] for res in results}
        # Return embeddings in the original order if possible, or just the list
        ordered_embeddings = [embedding_map.get(mid) for mid in int_movie_ids if mid in embedding_map]
        return ordered_embeddings
    except Exception as e:
        logging.error(f"Error querying embeddings by IDs: {e}")
        return []


def calculate_user_similarity(target_user_id, all_user_profiles):
    """Calculates Jaccard similarity between the target user and all other users."""
    if target_user_id not in all_user_profiles:
        return []

    target_prefs = all_user_profiles[target_user_id].get("preferences", set())
    if not target_prefs:
        return [] # Cannot calculate similarity without preferences

    similarities = []
    for other_user_id, profile in all_user_profiles.items():
        if other_user_id == target_user_id:
            continue

        other_prefs = profile.get("preferences", set())
        if not other_prefs:
            continue

        intersection = len(target_prefs.intersection(other_prefs))
        union = len(target_prefs.union(other_prefs))

        if union == 0:
            similarity = 0
        else:
            similarity = intersection / union

        if similarity > 0: # Only consider users with some overlap
            similarities.append({"user_id": other_user_id, "score": similarity})

    # Sort by similarity score descending
    similarities.sort(key=lambda x: x["score"], reverse=True)
    return similarities


def get_collaborative_recommendations(target_user_id, all_user_profiles, top_n_users=10, min_similarity=0.1):
    """Generates recommendations based on similar users' preferences (User-Based CF)."""
    target_profile = all_user_profiles.get(target_user_id)
    if not target_profile:
        return {} # Return empty dict

    target_prefs = target_profile.get("preferences", set())
    similar_users = calculate_user_similarity(target_user_id, all_user_profiles)

    # Aggregate recommendations from similar users
    item_scores = defaultdict(float)
    users_contributed = defaultdict(int)

    for sim_user_info in similar_users[:top_n_users]:
        user_id = sim_user_info["user_id"]
        similarity_score = sim_user_info["score"]

        if similarity_score < min_similarity:
            continue # Skip users below minimum similarity threshold

        other_prefs = all_user_profiles[user_id].get("preferences", set())

        for item_id in other_prefs:
            if item_id not in target_prefs: # Recommend only items not already preferred by target user
                item_scores[item_id] += similarity_score
                users_contributed[item_id] += 1

    # Convert scores dictionary to a list of {"id": item_id, "score": score}
    collab_recs_list = [{"id": item_id, "score": item_scores[item_id]} for item_id in item_scores]

    # Sort recommendations by score
    collab_recs_list.sort(key=lambda x: x["score"], reverse=True)

    return collab_recs_list


def get_user_recommendations(user_id, collection, top_k=10, content_weight=0.6, collab_weight=0.4):
    """
    Generates movie recommendations using a HYBRID approach:
    Content-Based (average embedding) + Collaborative Filtering (user-based).
    Uses Rank Aggregation (Borda Count variation).
    """
    profile = get_or_create_user(user_id)
    preferences = profile.get("preferences", set())

    if not preferences:
        logging.info(f"User {user_id} has no preferences. Cannot generate hybrid recommendations.")
        # Optionally return purely popular items or random items here
        return []

    logging.info(f"Generating HYBRID recommendations for user {user_id} based on {len(preferences)} preferences.")

    # --- 1. Content-Based Component ---
    content_recs = []
    preferred_embeddings = get_movie_embeddings_by_ids(collection, list(preferences), SEARCH_FIELD)
    if preferred_embeddings:
        avg_embedding = np.mean(np.array(preferred_embeddings), axis=0).tolist()
        # Fetch more initially to allow for merging/filtering
        content_recs_raw = search_similar_movies(
            collection,
            avg_embedding,
            search_field=SEARCH_FIELD,
            top_k=top_k * 2, # Fetch more for ranking
            exclude_ids=preferences
        )
        # Keep only IDs and a score (higher is better, use 1/(1+dist))
        content_recs = [{"id": r["id"], "score": 1.0 / (1.0 + r["distance"])} for r in content_recs_raw]
        logging.info(f"Content-Based component found {len(content_recs)} candidates.")
    else:
        logging.warning(f"Could not retrieve embeddings for user {user_id}'s preferences for content-based part.")


    # --- 2. Collaborative Filtering Component ---
    # Pass the global user_profiles dictionary (or fetch from DB if using one)
    collab_recs = get_collaborative_recommendations(user_id, user_profiles, top_n_users=20)
    logging.info(f"Collaborative Filtering component found {len(collab_recs)} candidates.")

    # --- 3. Hybridization (Rank Aggregation - Borda Count Style) ---
    final_scores = defaultdict(float)
    max_rank_points = top_k # Assign points based on rank

    # Assign points for content-based ranks
    for rank, rec in enumerate(content_recs[:top_k]): # Consider top_k from content
        points = max_rank_points - rank
        final_scores[rec['id']] += content_weight * points # Weighted points

    # Assign points for collaborative filtering ranks
    for rank, rec in enumerate(collab_recs[:top_k]): # Consider top_k from collab
        points = max_rank_points - rank
        final_scores[rec['id']] += collab_weight * points # Weighted points

    # Sort final recommendations by aggregated score
    sorted_hybrid_ids = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)

    # --- 4. Fetch Details for Top K Hybrid Results ---
    final_recommendations = []
    if sorted_hybrid_ids:
        top_hybrid_ids = sorted_hybrid_ids[:top_k]
        logging.info(f"Top {len(top_hybrid_ids)} hybrid recommendation IDs: {top_hybrid_ids}")

        # Query Milvus to get full details for the final list
        if top_hybrid_ids:
            try:
                expr = f"id in {top_hybrid_ids}"
                results = collection.query(
                    expr=expr,
                    output_fields=["id", "original_title", "overview", "cast", "director", "genres", "release_date"]
                )
                # Create a map for easy lookup
                results_map = {res['id']: res for res in results}
                # Build final list in the sorted order
                for movie_id in top_hybrid_ids:
                    if movie_id in results_map:
                        movie_details = results_map[movie_id]
                        # Add the hybrid score for potential display/debugging
                        movie_details['hybrid_score'] = final_scores[movie_id]
                        final_recommendations.append(movie_details)

            except Exception as e:
                logging.error(f"Error fetching details for hybrid recommendations: {e}")
                # Fallback or return empty might be needed

    logging.info(f"Generated {len(final_recommendations)} final hybrid recommendations for user {user_id}.")
    return final_recommendations


# --- Initial Setup ---
def initial_setup():
    """Connects to Milvus, creates collection, and loads initial data if needed."""
    if not connect_to_milvus():
        return None, None

    collection = get_milvus_collection()
    if collection is None: # Check if collection retrieval/creation failed
        logging.error("Failed to get or create Milvus collection.")
        return None, None

    # Check if collection is empty only after ensuring it's loaded
    try:
        if collection.is_empty: # Use is_empty property if available and collection is loaded
            logging.info("Collection is empty. Performing initial data load...")
            df = load_and_prepare_data(DATA_PATH)
            if df is not None:
                # Load a subset initially
                df_subset = df.head(1000) # Use the value from the selection
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
    except Exception as e:
        logging.error(f"Error checking collection status or loading initial data: {e}")
        # Decide how to proceed, maybe return None

    return collection, get_embedding_model()


if __name__ == "__main__":
    logging.info("Running main.py directly (for testing)...")
    collection, model = initial_setup()

    if collection and model:
        logging.info("Initial setup successful.")

        # --- Example User Profile Usage ---
        user_id_1 = "user123"
        user_id_2 = "user456"

        # Create users
        get_or_create_user(user_id_1, name="Alice")
        get_or_create_user(user_id_2, name="Bob")

        # Assume we have some movie IDs from insertion or previous knowledge
        # Let's try adding a new movie and getting its ID
        new_movie = {
            'original_title': 'Test Movie Alpha',
            'overview': 'A movie about testing recommendations.',
            'cast': 'AI Actor 1, AI Actor 2',
            'director': 'AI Director',
            'tagline': 'Will it blend?',
            'genres': 'Sci-Fi, Test',
            'release_date': '2025-04-23'
        }
        new_movie_id, _ = add_new_movie(collection, new_movie) # Get the ID

        # Add preferences (using the new ID if available, otherwise use placeholder IDs)
        # You might need to query existing IDs if you don't have them readily available
        # Example: query_result = collection.query(expr="id > 0", limit=2, output_fields=["id"])
        # existing_ids = [item['id'] for item in query_result]

        if new_movie_id:
            add_movie_preference(user_id_1, new_movie_id)
            add_movie_preference(user_id_2, new_movie_id) # Bob likes it too

        # Add some other hypothetical preferences (replace with actual IDs from your collection)
        # You can get IDs by querying: collection.query(expr = "original_title == 'Inception'", output_fields=["id"])
        # For this example, let's assume IDs 1695115316039704577 and 1695115316039704578 exist
        # Note: These IDs are placeholders from auto_id and will be different in your run.
        # Query your collection to get valid IDs for testing.
        try:
            query_res = collection.query(expr="id > 0", limit=2, output_fields=["id"])
            if len(query_res) >= 2:
                add_movie_preference(user_id_1, query_res[0]['id'])
                add_movie_preference(user_id_2, query_res[1]['id'])
            else:
                logging.warning("Could not retrieve enough existing movie IDs for preference testing.")
        except Exception as e:
            logging.error(f"Failed to query existing movie IDs: {e}")


        print("\n--- User Profiles ---")
        print(user_profiles)

        # Get recommendations for Alice
        print(f"\n--- Recommendations for {user_profiles[user_id_1]['name']} ---")
        alice_recs = get_user_recommendations(user_id_1, collection, top_k=3)
        if alice_recs:
            for rec in alice_recs:
                print(f"  - {rec['title']} (ID: {rec['id']}, Distance: {rec['distance']:.4f})")
        else:
            print("  No recommendations found.")

        # Get recommendations for Bob
        print(f"\n--- Recommendations for {user_profiles[user_id_2]['name']} ---")
        bob_recs = get_user_recommendations(user_id_2, collection, top_k=3)
        if bob_recs:
            for rec in bob_recs:
                print(f"  - {rec['title']} (ID: {rec['id']}, Distance: {rec['distance']:.4f})")
        else:
            print("  No recommendations found.")

    else:
        logging.error("Setup failed. Cannot run examples.")

    # connections.disconnect("default") # Disconnect if running as a standalone script
    # logging.info("Disconnected from Milvus.")