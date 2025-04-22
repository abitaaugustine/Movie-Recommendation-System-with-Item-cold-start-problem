from flask import Blueprint, request, jsonify, render_template
from .recommender import MovieRecommender
from .milvus_utils import MilvusManager

main_bp = Blueprint('main', __name__)
recommender = MovieRecommender()
milvus = MilvusManager()


@main_bp.route('/')
def index():
    return render_template('index.html')


@main_bp.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    return jsonify(recommender.recommend_similar_movies(data))


# @main_bp.route('/recommend_by_form', methods=['POST'])
# def recommend_form():
#     data = request.form
#     return jsonify(recommender.recommend_by_form(data))


@main_bp.route('/add_movie', methods=['POST'])
def add_movie():
    data = request.json
    milvus.insert_movies(data)
    return jsonify({"message": "Movie inserted successfully"})