import logging
from flask import Blueprint, render_template, request, jsonify
from app.services.get_query import get_user_profile_embedding, get_recommendations
# from app.services.get_recommendations import get_recommentations_service

logger = logging.getLogger(__name__)
index_bp = Blueprint("index_bp", __name__)

@index_bp.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@index_bp.route('/results')
def results():
    return render_template('results.html')


@index_bp.route("/query", methods=["POST"])
def query():
    data_request = request.get_json()
    resume_text = data_request.get("resume")
    personal_interests = data_request.get("interests")
    threshold = data_request.get("threshold", 0.33)
    top_n = data_request.get("top_n", 20)

    # Obter o embedding do perfil do usuário
    user_embedding = get_user_profile_embedding(resume_text, personal_interests)

    # Obter recomendações
    recommendations = get_recommendations(user_embedding, threshold, top_n)

    return jsonify(recommendations)
