from flask import Flask
from .milvus_utils import init_milvus


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'supersecretkey'  # Change in production

    init_milvus()  # Initialize Milvus connection and collection setup

    from .routes import main_bp
    app.register_blueprint(main_bp)

    return app