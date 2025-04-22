from flask import Flask, request, render_template, flash, redirect, url_for
import main as mrs # Import our movie recommendation system logic
import os

app = Flask(__name__)
# Required for flashing messages
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

@app.route('/', methods=['GET'])
def index():
    """Renders the main page."""
    return render_template('index.html', recommendations=None, add_results=None, new_movie_title=None)

@app.route('/recommend', methods=['POST'])
def recommend():
    """Handles movie recommendation requests."""
    if not collection:
        flash("Error: Milvus collection not available. Setup might have failed.", "danger")
        return redirect(url_for('index'))

    try:
        # Get data from form
        title_query = request.form.get('title_query', '')
        overview_query = request.form.get('overview_query', '')
        cast_query = request.form.get('cast_query', '')
        director_query = request.form.get('director_query', '')
        tagline_query = request.form.get('tagline_query', '')
        genres_query = request.form.get('genres_query', '')
        top_k = int(request.form.get('top_k', 5))

        # Combine inputs for a single query embedding
        query_text = f"{title_query}. {overview_query}. Cast: {cast_query}. Director: {director_query}. Genres: {genres_query}. Tagline: {tagline_query}"

        if not query_text.strip('.'): # Check if query is effectively empty
             flash("Please provide some details to search for.", "warning")
             return redirect(url_for('index'))

        print(f"Generating embedding for query: {query_text[:100]}...") # Log snippet
        query_embedding = mrs.generate_embeddings([query_text])

        recommendations = []
        if query_embedding:
            print(f"Searching similar movies using field: {mrs.SEARCH_FIELD}")
            recommendations = mrs.search_similar_movies(collection, query_embedding[0], search_field=mrs.SEARCH_FIELD, top_k=top_k)
            print(f"Found {len(recommendations)} recommendations.")
            if not recommendations:
                 flash("No similar movies found for your query.", "info")
        else:
            flash("Error generating embedding for your query.", "danger")

        return render_template('index.html', recommendations=recommendations, add_results=None, new_movie_title=None)

    except Exception as e:
        print(f"Error during recommendation: {e}")
        flash(f"An error occurred during recommendation: {e}", "danger")
        return redirect(url_for('index'))


@app.route('/add', methods=['POST'])
def add():
    """Handles adding a new movie."""
    if not collection:
        flash("Error: Milvus collection not available. Setup might have failed.", "danger")
        return redirect(url_for('index'))

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
            results = mrs.search_similar_movies(collection, primary_embedding, search_field=mrs.SEARCH_FIELD, top_k=6)
            add_results = [res for res in results if res['id'] != new_id][:5] # Exclude self, limit 5
            print(f"Found {len(add_results)} similar movies for the new item.")
            if not add_results:
                 flash("No other similar movies found for the newly added item.", "info")

        else:
            flash(f"Failed to add movie '{new_title}'.", "danger")

        # Pass results back to the template
        return render_template('index.html', recommendations=None, add_results=add_results, new_movie_title=new_title if new_id else None)

    except Exception as e:
        print(f"Error adding movie: {e}")
        flash(f"An error occurred while adding the movie: {e}", "danger")
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Make sure templates folder exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Created 'templates' directory.")
        # You might want to create a basic index.html here if it doesn't exist
        if not os.path.exists('templates/index.html'):
             with open('templates/index.html', 'w') as f:
                 f.write('<html><head><title>Movie Recommender</title></head><body><h1>App Initializing...</h1></body></html>')
             print("Created basic 'templates/index.html'. Please replace with full template.")

    app.run(debug=True) # debug=True for development, set to False for production