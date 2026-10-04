
from flask import Flask, render_template, jsonify, abort
import json
import os
from ml.predict import analyze_project

app = Flask(__name__)

DATA_FILE = os.path.join("data", "projects.json")


def load_projects():
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/project/<project_id>")
def project_page(project_id):
    projects = load_projects()

    project = next(
        (p for p in projects if p["project_id"] == project_id),
        None
    )

    if not project:
        abort(404)

    return render_template("project.html", project=project)


@app.route("/api/projects")
def get_projects():
    return jsonify(load_projects())


@app.route("/api/projects/<project_id>")
def get_project(project_id):
    projects = load_projects()

    project = next(
        (p for p in projects if p["project_id"] == project_id),
        None
    )

    if not project:
        return jsonify({"error": "Project not found"}), 404

    return jsonify(project)


@app.route("/api/summary")
def summary():
    projects = load_projects()

    total = len(projects)

    high_risk = sum(
        1 for p in projects
        if p["delay_risk"] >= 70
    )

    medium_risk = sum(
        1 for p in projects
        if 40 <= p["delay_risk"] < 70
    )

    low_risk = sum(
        1 for p in projects
        if p["delay_risk"] < 40
    )

    total_budget = sum(
        p["budget"] for p in projects
    )

    total_spent = sum(
        p["spent"] for p in projects
    )

    return jsonify({
        "total_projects": total,
        "high_risk": high_risk,
        "medium_risk": medium_risk,
        "low_risk": low_risk,
        "total_budget": total_budget,
        "total_spent": total_spent
    })

@app.route("/api/predict/<project_id>")
def predict_project(project_id):

    projects = load_projects()

    project = next(
        (
            p for p in projects
            if p["project_id"] == project_id
        ),
        None
    )

    if not project:
        return jsonify({
            "error": "Project not found"
        }), 404

    result = analyze_project(
        project
    )

    return jsonify(result)


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )

