from flask import Flask, request, render_template, flash, redirect, url_for, session # Added session
import main as mrs # Import our movie recommendation system logic
import os

app = Flask(__name__)
# Required for flashing messages and session management
app.secret_key = os.urandom(24) # Replace with a strong secret key in production

# --- Global Variables ---
# Initialize collection and model (call initial_setup once)
# In a production app, consider better ways to manage global state or app context
print("Performing initial setup...")
collection, model = mrs.initial_setup()
if not collection or not model:
    print("FATAL: Milvus connection or model loading failed. Flask app cannot start properly.")
    # In a real app, you might exit or raise an exception
    # For now, we'll let it run but endpoints might fail.
    collection = None # Ensure it's None if setup failed

print("Initial setup complete.")

# --- Available Search Facets ---
# Make embed fields available for dropdown (map internal name to user-friendly name)
AVAILABLE_FACETS = {
    "overview": "Overview",
    "title": "Title",
    "cast": "Cast",
    "genres": "Genres",
    # Add a combined option if desired, though current implementation uses a single combined query text
    # "combined": "Combined Text"
}
DEFAULT_FACET_KEY = "overview" # Corresponds to mrs.SEARCH_FIELD usually

# --- Helper Function ---
def get_movie_details_by_ids(movie_ids):
    """Fetches basic movie details (ID, title) for a list of IDs."""
    if not collection or not movie_ids:
        return {}
    try:
        # Ensure IDs are integers
        int_movie_ids = [int(mid) for mid in movie_ids]
        expr = f"id in {list(int_movie_ids)}"
        results = collection.query(
            expr=expr,
            output_fields=["id", "original_title"] # Only fetch needed fields
        )
        # Create a dictionary mapping ID to title
        details_map = {res['id']: res['original_title'] for res in results}
        return details_map
    except Exception as e:
        print(f"Error querying movie details by IDs: {e}")
        return {}

# --- Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main page."""
    # Simple user management: Use session to remember the user ID
    user_id = session.get('user_id', 'guest') # Default to 'guest' if not logged in
    return render_template('index.html',
                           recommendations=None,
                           add_results=None,
                           new_movie_title=None,
                           user_id=user_id, # Pass user_id to template
                           available_facets=AVAILABLE_FACETS, # Pass facets to template
                           selected_facet=DEFAULT_FACET_KEY) # Pass default facet


@app.route('/set_user', methods=['POST'])
def set_user():
    """Sets the user ID in the session."""
    user_id = request.form.get('user_id')
    if user_id:
        session['user_id'] = user_id.strip()
        flash(f"User set to: {session['user_id']}", "info")
        # Ensure user profile exists in our simple store
        mrs.get_or_create_user(session['user_id'])
    else:
        flash("Please enter a User ID.", "warning")
    return redirect(url_for('index'))


@app.route('/recommend', methods=['POST'])
def recommend():
    """Handles movie recommendation requests."""
    if not collection:
        flash("Error: Milvus collection not available. Setup might have failed.", "danger")
        return redirect(url_for('index'))

    user_id = session.get('user_id', 'guest') # Get current user

    try:
        # Get data from form
        title_query = request.form.get('title_query', '')
        overview_query = request.form.get('overview_query', '')
        cast_query = request.form.get('cast_query', '')
        director_query = request.form.get('director_query', '')
        tagline_query = request.form.get('tagline_query', '')
        genres_query = request.form.get('genres_query', '')
        top_k = int(request.form.get('top_k', 5))
        search_facet_key = request.form.get('search_facet', DEFAULT_FACET_KEY) # Get selected facet

        # Validate selected facet
        if search_facet_key not in AVAILABLE_FACETS:
            flash(f"Invalid search facet selected. Defaulting to {AVAILABLE_FACETS[DEFAULT_FACET_KEY]}.", "warning")
            search_facet_key = DEFAULT_FACET_KEY

        # Determine the embedding field to search against in Milvus
        # Assumes facet key matches the start of the embedding field name (e.g., 'overview' -> 'overview_embedding')
        search_field_milvus = f"{search_facet_key}_embedding"
        # Double-check if this field actually exists in the schema/mrs config if necessary

        # Determine the primary text input based on the facet for embedding generation
        # (Could also combine all text, but searching a specific facet implies using that facet's text primarily)
        query_text = ""
        if search_facet_key == 'title':
            query_text = title_query
        elif search_facet_key == 'cast':
            query_text = cast_query
        elif search_facet_key == 'genres':
            query_text = genres_query
        # Default to overview or combine if preferred
        # elif search_facet_key == 'overview':
        #     query_text = overview_query
        else: # Default to overview or a combination if overview is empty
            query_text = overview_query if overview_query else f"{title_query} {cast_query} {genres_query}"


        # Ensure there is *some* text to search with, even if not the primary facet field
        if not query_text:
            query_text = f"{title_query} {overview_query} {cast_query} {director_query} {genres_query} {tagline_query}"


        if not query_text.strip('.'):
            flash("Please provide some details to search for.", "warning")
            # Need to pass facets back even on redirect/error render
            return render_template('index.html',
                                    recommendations=None, add_results=None, new_movie_title=None,
                                    user_id=user_id, available_facets=AVAILABLE_FACETS,
                                    selected_facet=search_facet_key)


        print(f"Generating embedding for query (focused on {search_facet_key}): {query_text[:100]}...")
        query_embedding = mrs.generate_embeddings([query_text])

        recommendations = []
        if query_embedding:
            print(f"Searching similar movies using field: {search_field_milvus}")
            recommendations = mrs.search_similar_movies(
                collection, 
                query_embedding[0], 
                search_field=search_field_milvus, # Use the facet-specific field
                top_k=top_k
            )
            print(f"Found {len(recommendations)} recommendations based on {AVAILABLE_FACETS[search_facet_key]}.")
            if not recommendations:
                flash(f"No similar movies found based on {AVAILABLE_FACETS[search_facet_key]} for your query.", "info")
        else:
            flash("Error generating embedding for your query.", "danger")

        # Pass results and selected facet back to template
        return render_template('index.html',
                               recommendations=recommendations,
                               add_results=None,
                               new_movie_title=None,
                               user_id=user_id, # Pass user_id
                               available_facets=AVAILABLE_FACETS, # Pass facets again
                               selected_facet=search_facet_key) # Pass selected facet

    except Exception as e:
        print(f"Error during recommendation: {e}")
        flash(f"An error occurred during recommendation: {e}", "danger")
        # Need to pass facets back even on redirect/error render
        return render_template('index.html',
                               recommendations=None, add_results=None, new_movie_title=None,
                               user_id=user_id, available_facets=AVAILABLE_FACETS,
                               selected_facet=request.form.get('search_facet', DEFAULT_FACET_KEY))


@app.route('/add', methods=['POST'])
def add():
    """Handles adding a new movie."""
    if not collection:
        flash("Error: Milvus collection not available. Setup might have failed.", "danger")
        return redirect(url_for('index'))

    user_id = session.get('user_id', 'guest') # Get current user

    try:
        # Get data from form
        new_title = request.form.get('new_title')
        new_overview = request.form.get('new_overview')

        if not new_title or not new_overview:
            flash("Title and Overview are required fields.", "warning")
            return redirect(url_for('index'))

        movie_data = {
            "original_title": new_title,
            "overview": new_overview,
            "cast": request.form.get('new_cast', ''),
            "director": request.form.get('new_director', ''),
            "tagline": request.form.get('new_tagline', ''),
            "genres": request.form.get('new_genres', ''),
            "release_date": request.form.get('new_release_date', '') # Assuming YYYY-MM-DD format
        }

        print(f"Adding new movie: {new_title}")
        new_id, primary_embedding = mrs.add_new_movie(collection, movie_data)

        add_results = []
        if new_id and primary_embedding:
            flash(f"Successfully added '{new_title}' (ID: {new_id}). Finding similar movies...", "success")
            print(f"Searching similar movies for the new item using field: {mrs.SEARCH_FIELD}")
            # Search for movies similar to the newly added one (excluding itself)
            search_field_after_add = mrs.SEARCH_FIELD
            print(f"Searching similar movies for the new item using field: {search_field_after_add}")
            results = mrs.search_similar_movies(collection, primary_embedding, search_field=search_field_after_add, top_k=6)
            add_results = [res for res in results if res['id'] != new_id][:5] # Exclude self, limit 5
            print(f"Found {len(add_results)} similar movies for the new item.")
            if not add_results:
                flash("No other similar movies found for the newly added item.", "info")

        else:
            flash(f"Failed to add movie '{new_title}'.", "danger")

        # Pass results back to the template
        return render_template('index.html',
                               recommendations=None,
                               add_results=add_results,
                               new_movie_title=new_title if new_id else None,
                               user_id=user_id, # Pass user_id
                               available_facets=AVAILABLE_FACETS, # Pass facets
                               selected_facet=DEFAULT_FACET_KEY) # Pass default facet

    except Exception as e:
        print(f"Error adding movie: {e}")
        flash(f"An error occurred while adding the movie: {e}", "danger")
        return redirect(url_for('index'))


# --- User Profile Routes ---
@app.route('/profile')
def profile():
    """Displays the current user's profile and preferences."""
    user_id = session.get('user_id')
    if not user_id:
        flash("Please set a User ID first.", "warning")
        return redirect(url_for('index'))

    if not collection:
        flash("Error: Milvus collection not available.", "danger")
        return redirect(url_for('index'))

    user_profile = mrs.get_or_create_user(user_id)
    preferences = user_profile.get("preferences", set())

    # Get details for preferred movies
    preferred_movie_details = {}
    if preferences:
        preferred_movie_details = get_movie_details_by_ids(list(preferences))

    # Prepare preferences list for template (ID and Title)
    preferences_list = [
        {"id": mid, "title": preferred_movie_details.get(mid, "Unknown Title")}
        for mid in preferences
    ]

    return render_template('profile.html',
                           user_id=user_id,
                           user_profile=user_profile,
                           preferences_list=preferences_list,
                           user_recommendations=None) # Initially no recommendations shown


@app.route('/add_preference', methods=['POST'])
def add_preference():
    """Adds a movie to the current user's preferences."""
    user_id = session.get('user_id')
    if not user_id:
        flash("Please set a User ID before adding preferences.", "warning")
        return redirect(request.referrer or url_for('index')) # Redirect back

    movie_id = request.form.get('movie_id')
    movie_title = request.form.get('movie_title', 'this movie') # Get title for flash message

    if not movie_id:
        flash("Invalid movie ID.", "danger")
        return redirect(request.referrer or url_for('index'))

    try:
        # Convert movie_id to int if your main.py expects it
        movie_id_int = int(movie_id)
        added = mrs.add_movie_preference(user_id, movie_id_int)
        if added:
            flash(f"Added '{movie_title}' (ID: {movie_id_int}) to your preferences.", "success")
        else:
            flash(f"'{movie_title}' (ID: {movie_id_int}) is already in your preferences.", "info")
    except ValueError:
         flash("Invalid movie ID format.", "danger")
    except Exception as e:
        print(f"Error adding preference: {e}")
        flash(f"An error occurred while adding preference: {e}", "danger")

    # Redirect back to the page the user came from (e.g., search results)
    return redirect(url_for('index'))


@app.route('/user_recommendations')
def user_recommendations():
    """Generates and displays personalized HYBRID recommendations for the current user."""
    user_id = session.get('user_id')
    if not user_id:
        flash("Please set a User ID first.", "warning")
        return redirect(url_for('index'))

    if not collection:
        flash("Error: Milvus collection not available.", "danger")
        return redirect(url_for('profile')) # Redirect to profile page on error

    try:
        top_k = 5 # Or get from request args: int(request.args.get('top_k', 5))
        print(f"Getting HYBRID recommendations for user: {user_id}")
        recommendations = mrs.get_user_recommendations(user_id, collection, top_k=top_k)
        print(f"Found {len(recommendations)} user recommendations.")

        if not recommendations:
            flash("Could not generate recommendations. Add more preferences or explore general search!", "info")
            # Still render profile page, but without recommendations section
            return redirect(url_for('profile')) # Redirect back to profile

        # Need profile data again for the profile template
        user_profile = mrs.get_or_create_user(user_id)
        preferences = user_profile.get("preferences", set())
        preferred_movie_details = get_movie_details_by_ids(list(preferences))
        preferences_list = [
            {"id": mid, "title": preferred_movie_details.get(mid, "Unknown Title")}
            for mid in preferences
        ]

        # Render the profile page, now including the recommendations
        return render_template('profile.html',
                               user_id=user_id,
                               user_profile=user_profile,
                               preferences_list=preferences_list,
                               user_recommendations=recommendations) # Pass recommendations

    except Exception as e:
        print(f"Error getting user recommendations: {e}")
        flash(f"An error occurred while getting recommendations: {e}", "danger")
        return redirect(url_for('profile'))


if __name__ == '__main__':
    # Make sure templates folder exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Created 'templates' directory.")
        # You might want to create a basic index.html here if it doesn't exist
        if not os.path.exists('templates/index.html'):
            with open('templates/index.html', 'w') as f:
                f.write('<html><head><title>Movie Recommender</title></head><body><h1>App Initializing...</h1><p>Create templates/index.html and templates/profile.html</p></body></html>')
            print("Created basic 'templates/index.html'. Please replace with full template.")
        if not os.path.exists('templates/profile.html'):
            with open('templates/profile.html', 'w') as f:
                f.write('<html><head><title>User Profile</title></head><body><h1>User Profile Page</h1><p>Create templates/profile.html</p></body></html>')
            print("Created basic 'templates/profile.html'. Please replace with full template.")


    app.run(debug=True) # debug=True for development, set to False for production
