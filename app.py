from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
from utils import skill_tokenizer

app = Flask(__name__)
CORS(app)

# Load trained model and artifacts
model = joblib.load("model/skill_model.pkl")

vectorizer = joblib.load("model/vectorizer.pkl")

with open("model/skills_dict.json", "r") as f:
    skills_dict = json.load(f)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Skill Gap ML service running"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Input:  { "user_skills": ["Python", "React", "SQL"], "target_role": "Data Scientist" }
    Output: { "target_role", "match_score", "missing_skills", "matched_skills", "recommendations" }
    """
    data = request.get_json()
    user_skills_raw = data.get("user_skills", [])
    target_role = data.get("target_role", "")

    if not user_skills_raw:
        return jsonify({"error": "user_skills is required"}), 400

    # Normalize
    user_skills = [s.strip().lower() for s in user_skills_raw]
    user_skills_text = ", ".join(user_skills)

    # If no target role, predict the best matching role
    if not target_role:
        vec = vectorizer.transform([user_skills_text])
        probabilities = model.predict_proba(vec)[0]
        top_indices = np.argsort(probabilities)[::-1][:3]
        classes = model.classes_
        suggestions = [
            {"role": classes[i], "confidence": round(float(probabilities[i]) * 100, 1)}
            for i in top_indices
        ]
        target_role = classes[top_indices[0]]
    else:
        suggestions = []

    # Skill gap analysis
    if target_role not in skills_dict:
        return jsonify({"error": f"Role '{target_role}' not found"}), 404

    required_skills = [s.lower() for s in skills_dict[target_role]]
    matched_skills = [s for s in required_skills if s in user_skills]
    missing_skills = [s for s in required_skills if s not in user_skills]

    match_score = round((len(matched_skills) / len(required_skills)) * 100, 1)

    # Priority categorize missing skills
    total_required = len(required_skills)
    recommendations = []
    for i, skill in enumerate(missing_skills):
        priority = "High" if i < 3 else ("Medium" if i < 6 else "Low")
        recommendations.append({
            "skill": skill.title(),
            "priority": priority,
            "learning_time": estimate_learning_time(skill)
        })

    return jsonify({
        "target_role": target_role,
        "match_score": match_score,
        "matched_skills": [s.title() for s in matched_skills],
        "missing_skills": [s.title() for s in missing_skills],
        "total_required": total_required,
        "role_suggestions": suggestions,
        "recommendations": recommendations
    })


@app.route("/roles", methods=["GET"])
def get_roles():
    """Return all supported roles"""
    return jsonify({"roles": list(skills_dict.keys())})


@app.route("/skills/<role>", methods=["GET"])
def get_skills_for_role(role):
    """Return required skills for a specific role"""
    if role not in skills_dict:
        return jsonify({"error": "Role not found"}), 404
    return jsonify({"role": role, "skills": skills_dict[role]})


def estimate_learning_time(skill):
    complex_skills = ["machine learning", "deep learning", "kubernetes", "tensorflow",
                      "pytorch", "blockchain", "solidity", "fpga", "assembly"]
    medium_skills = ["docker", "aws", "react", "python", "sql", "node.js", "typescript",
                     "mongodb", "redis", "graphql"]
    if skill.lower() in complex_skills:
        return "3-6 months"
    elif skill.lower() in medium_skills:
        return "1-3 months"
    else:
        return "2-4 weeks"


if __name__ == "__main__":
    app.run(debug=True, port=5001)
