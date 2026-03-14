from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    g,
    send_from_directory,
)
import pandas as pd
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3
import os
import json
from io import StringIO

app = Flask(__name__)
app.secret_key = "careersync_secret_2026_secure_key"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB
app.config["SESSION_PERMANENT"] = False

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

DATABASE = "database.db"

skills_df = pd.read_csv("skills_master.csv")
skills_df.columns = skills_df.columns.str.strip().str.lower()

jobs = pd.read_csv("job_openings.csv")
courses = pd.read_csv("courses.csv")

jobs.columns = jobs.columns.str.strip().str.lower()
courses.columns = courses.columns.str.strip().str.lower()


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = get_db()

    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    """)

    columns = [row["name"] for row in db.execute("PRAGMA table_info(users)").fetchall()]
    if "is_admin" not in columns:
        db.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")

    db.execute("""
        CREATE TABLE IF NOT EXISTS applied_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            applicant_name TEXT,
            title TEXT,
            company TEXT,
            location TEXT,
            apply_link TEXT,
            resume_filename TEXT,
            status TEXT DEFAULT 'Applied',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, title, company, apply_link)
        )
    """)

    db.commit()

    # Automatically make the first registered user admin
    admin_exists = db.execute(
        "SELECT * FROM users WHERE is_admin = 1"
    ).fetchone()

    if not admin_exists:
        first_user = db.execute(
            "SELECT id FROM users ORDER BY id ASC LIMIT 1"
        ).fetchone()

        if first_user:
            db.execute(
                "UPDATE users SET is_admin = 1 WHERE id = ?",
                (first_user["id"],)
            )
            db.commit()


@app.before_request
def setup():
    init_db()


def login_required():
    return "user_id" in session


def admin_required():
    return login_required() and session.get("is_admin") == 1


def normalize_skill(skill):
    return skill.strip().lower().replace("-", " ").replace("_", " ")


def parse_skills(skills_text):
    return [normalize_skill(skill) for skill in skills_text.split(",") if skill.strip()]


def extract_skills_from_resume_path(file_path):
    text = extract_text(file_path)
    text = text.lower().replace("\n", " ")

    skill_keywords = skills_df["skill"].dropna().tolist()
    skill_keywords = [normalize_skill(skill) for skill in skill_keywords]

    extracted_skills = []
    for skill in skill_keywords:
        if skill in text:
            extracted_skills.append(skill)

    return list(set(extracted_skills))


def match_jobs(user_skills):
    jobs_copy = jobs.copy()
    user_skills = [normalize_skill(skill) for skill in user_skills]

    job_skills = jobs_copy["skills"].tolist()
    all_skills = job_skills + [", ".join(user_skills)]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_skills)

    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    jobs_copy["match_score"] = similarity[0]

    sorted_jobs = jobs_copy.sort_values(by="match_score", ascending=False)
    positive_matches = sorted_jobs[sorted_jobs["match_score"] > 0.15]

    results = positive_matches.drop_duplicates(subset=["title"]).head(5).reset_index(drop=True)

    matched_titles = results["title"].tolist()
    openings = positive_matches[positive_matches["title"].isin(matched_titles)].sort_values(
        by=["title", "match_score"], ascending=[True, False]
    ).reset_index(drop=True)

    missing_skills_list = []

    for job in results.itertuples():
        required = [normalize_skill(skill) for skill in parse_skills(job.skills)]
        missing = [skill for skill in required if skill not in user_skills]
        missing_skills_list.append(missing)

    return results, openings, missing_skills_list


def recommend_courses_flat(missing_skills_list):
    all_missing = set()

    for skills in missing_skills_list:
        for skill in skills:
            all_missing.add(skill.strip().lower())

    matched_courses = []

    for skill in all_missing:
        rows = courses[courses["skill"].str.strip().str.lower() == skill]
        if not rows.empty:
            matched_courses.append(rows)

    if matched_courses:
        recommended_courses = pd.concat(matched_courses, ignore_index=True)
        recommended_courses = recommended_courses.drop_duplicates(subset=["skill", "course"])
        recommended_courses = recommended_courses.sort_values(by="skill")
        return recommended_courses.to_dict(orient="records")

    return []


@app.route("/")
def home():
    if login_required():
        if "results" in session:
            return redirect(url_for("job_matches"))
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        if not full_name or not email or not password:
            error = "Please fill all fields."
            return render_template("signup.html", error=error)

        username = email.split("@")[0].lower()

        db = get_db()

        existing_email = db.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,)
        ).fetchone()

        if existing_email:
            error = "Email already registered."
            return render_template("signup.html", error=error)

        existing_username = db.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        ).fetchone()

        if existing_username:
            error = "Username already exists. Try another email."
            return render_template("signup.html", error=error)

        password_hash = generate_password_hash(password, method="pbkdf2:sha256")

        db.execute(
            "INSERT INTO users (full_name, email, username, password_hash) VALUES (?, ?, ?, ?)",
            (full_name, email, username, password_hash)
        )
        db.commit()

        return redirect(url_for("login", created=username))

    return render_template("signup.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "").strip()

        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        ).fetchone()

        if not user or not check_password_hash(user["password_hash"], password):
            error = "Invalid username or password."
            return render_template("login.html", error=error, created_user=None)

        session["user_id"] = user["id"]
        session["user_name"] = user["full_name"]
        session["username"] = user["username"]
        session["is_admin"] = user["is_admin"]
        session.permanent = False

        return redirect(url_for("dashboard"))

    created_user = request.args.get("created")
    return render_template("login.html", error=error, created_user=created_user)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if not login_required():
        return redirect(url_for("login"))

    error_message = None

    if request.method == "POST":
        user_skills = []

        resume = request.files.get("resume")
        pdf_uploaded = resume and resume.filename != ""

        if pdf_uploaded:
            safe_name = secure_filename(resume.filename)
            stored_filename = f"{session['user_id']}_{safe_name}"
            stored_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_filename)

            resume.save(stored_path)
            session["last_resume_filename"] = stored_filename

            user_skills = extract_skills_from_resume_path(stored_path)

        if not user_skills:
            skills_text = request.form.get("skills", "").strip()
            if skills_text:
                user_skills = parse_skills(skills_text)

        if not pdf_uploaded and not user_skills:
            error_message = "Please upload your resume or enter skills."
            return render_template(
                "dashboard.html",
                error=error_message,
                user_name=session.get("user_name")
            )

        has_detected_skills = len(user_skills) > 0
        job_results, job_openings, missing_skills = match_jobs(user_skills)
        course_recommendations = recommend_courses_flat(missing_skills)
        has_job_matches = not job_results.empty
        job_titles = job_openings["title"].tolist()
        total_jobs = len(job_openings)

        if not job_openings.empty:
            role_counts = job_openings["title"].value_counts()
            max_count = role_counts.max()
            top_roles = role_counts[role_counts == max_count].index.tolist()
            top_role = " • ".join(top_roles)

            salary_by_role = (
                job_openings.groupby("title")["salary"]
                .mean()
                .round()
                .astype(int)
                .to_dict()
            )
        else:
            top_role = "N/A"
            salary_by_role = {}

        openings_records = job_openings.to_dict(orient="records")

        session["results"] = job_results.to_json()
        session["openings"] = job_openings.to_json()
        session["missing_skills"] = json.dumps(missing_skills)
        session["courses"] = json.dumps(course_recommendations)
        session["user_skills"] = json.dumps(user_skills)
        session["job_titles"] = json.dumps(job_titles)
        session["total_jobs"] = total_jobs
        session["salary_by_role"] = json.dumps(salary_by_role)
        session["top_role"] = top_role

        return render_template(
            "result.html",
            results=job_results,
            openings=job_openings,
            openings_records=openings_records,
            missing_skills=missing_skills,
            course_recommendations=course_recommendations,
            user_skills=user_skills,
            job_titles=job_titles,
            total_jobs=total_jobs,
            salary_by_role=salary_by_role,
            top_role=top_role,
            has_detected_skills=has_detected_skills,
            has_job_matches=has_job_matches,
            user_name=session.get("user_name")
        )

    return render_template("dashboard.html", error=error_message, user_name=session.get("user_name"))


@app.route("/job-matches")
def job_matches():
    if not login_required():
        return redirect(url_for("login"))

    if "results" not in session:
        return redirect(url_for("dashboard"))

    results = pd.read_json(StringIO(session["results"]))
    openings = pd.read_json(StringIO(session["openings"]))
    missing_skills = json.loads(session["missing_skills"])
    courses = json.loads(session["courses"])
    user_skills = json.loads(session["user_skills"])
    job_titles = json.loads(session["job_titles"])
    total_jobs = session["total_jobs"]
    salary_by_role = json.loads(session["salary_by_role"])
    top_role = session["top_role"]

    openings_records = openings.to_dict(orient="records")

    return render_template(
        "result.html",
        results=results,
        openings=openings,
        openings_records=openings_records,
        missing_skills=missing_skills,
        course_recommendations=courses,
        user_skills=user_skills,
        job_titles=job_titles,
        total_jobs=total_jobs,
        salary_by_role=salary_by_role,
        top_role=top_role,
        has_detected_skills=len(user_skills) > 0,
        has_job_matches=not results.empty,
        user_name=session.get("user_name")
    )


@app.route("/save-job", methods=["POST"])
def save_job():
    if not login_required():
        return redirect(url_for("login"))

    resume_filename = session.get("last_resume_filename")
    title = request.form.get("title")
    company = request.form.get("company")
    location = request.form.get("location")
    apply_link = request.form.get("apply_link")

    if not title or not company or not apply_link:
        return redirect(url_for("job_matches"))

    db = get_db()
    db.execute("""
        INSERT OR IGNORE INTO applied_jobs
        (user_id, applicant_name, title, company, location, apply_link, resume_filename, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'Applied')
    """, (
        session["user_id"],
        session["user_name"],
        title,
        company,
        location,
        apply_link,
        resume_filename
    ))
    db.commit()

    return redirect(apply_link)


@app.route("/view-resume/<filename>")
def view_resume(filename):
    if not login_required():
        return redirect(url_for("login"))

    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/applied-jobs")
def applied_jobs():
    if not login_required():
        return redirect(url_for("login"))

    db = get_db()
    jobs_saved = db.execute("""
        SELECT * FROM applied_jobs
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (session["user_id"],)).fetchall()

    return render_template(
        "applied_jobs.html",
        jobs_saved=jobs_saved,
        user_name=session.get("user_name")
    )


@app.route("/admin")
def admin_dashboard():
    if not login_required():
        return redirect(url_for("login"))

    if not admin_required():
        return redirect(url_for("dashboard"))

    db = get_db()
    applications = db.execute("""
        SELECT * FROM applied_jobs
        ORDER BY created_at DESC
    """).fetchall()

    return render_template("admin.html", applications=applications)


@app.route("/approve/<int:app_id>")
def approve_application(app_id):
    if not login_required():
        return redirect(url_for("login"))

    if not admin_required():
        return redirect(url_for("dashboard"))

    db = get_db()
    db.execute("""
        UPDATE applied_jobs
        SET status = 'Approved'
        WHERE id = ?
    """, (app_id,))
    db.commit()
    return redirect(url_for("admin_dashboard"))


@app.route("/reject/<int:app_id>")
def reject_application(app_id):
    if not login_required():
        return redirect(url_for("login"))

    if not admin_required():
        return redirect(url_for("dashboard"))

    db = get_db()
    db.execute("""
        UPDATE applied_jobs
        SET status = 'Rejected'
        WHERE id = ?
    """, (app_id,))
    db.commit()
    return redirect(url_for("admin_dashboard"))

@app.route("/clear-results")
def clear_results():
    if not login_required():
        return redirect(url_for("login"))

    keys_to_remove = [
        "results",
        "openings",
        "missing_skills",
        "courses",
        "user_skills",
        "job_titles",
        "total_jobs",
        "salary_by_role",
        "top_role",
        "last_resume_filename"
    ]

    for key in keys_to_remove:
        session.pop(key, None)

    return redirect(url_for("dashboard"))

@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/auto-logout", methods=["POST"])
def auto_logout():
    session.clear()
    return ("", 204)

if __name__ == "__main__":
    app.run(debug=True)