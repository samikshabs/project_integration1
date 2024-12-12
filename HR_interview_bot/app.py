from flask import Flask, render_template, request, redirect, url_for, session
import mysql.connector
import os
import pandas as pd
from werkzeug.utils import secure_filename
from flask import Flask, request
import chromadb
import os
from werkzeug.utils import secure_filename
from models.rag_model import process_resume_and_match_jobs
from models.rag_model import generate_questions_for_job
from models.rag_model import get_question_answer_mapping
from models.rag_model import evaluate_answer

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Ready!4you",
    database="hr_bot_db"
)

# Initialize ChromaDB Client
chroma_client = chromadb.Client()

# Create or get the collection
collection_name = "company_data"
collection = chroma_client.get_or_create_collection(collection_name)

# Function to load company data into ChromaDB
def load_company_data_to_chromadb(file_path):
    """Load company details from an Excel file into ChromaDB if not already loaded."""
    # Read the data from Excel
    df = pd.read_excel(file_path, usecols=["jobTitle", "jobUrl", "jobDescription", "jobCompany"])
    
    # Fetch existing metadata to find already stored job titles
    existing_metadatas = collection.get(include=["metadatas"])["metadatas"]
    existing_ids = {metadata["jobTitle"] for metadata in existing_metadatas}

    # Insert data into ChromaDB only if it doesn't already exist
    new_records = 0
    for _, row in df.iterrows():
        if row["jobTitle"] not in existing_ids:  # Check if the jobTitle is already in the collection
            collection.add(
                documents=[row["jobDescription"]],  # Add the job description as the document
                metadatas=[{
                    "jobTitle": row["jobTitle"], 
                    "jobUrl": row["jobUrl"],
                    "jobCompany": row["jobCompany"]  # Add the company name to metadata
                }],  
                ids=[row["jobTitle"]]  # Use jobTitle as a unique identifier
            )
            new_records += 1

    print(f"{new_records} new records added to ChromaDB. Skipping duplicates.")

# Load the company data into ChromaDB (only once when the app is started)
load_company_data_to_chromadb('company_data.xlsx')
# Configure the upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Route for Home
@app.route('/')
def home():
    return render_template('home.html')

# Route for Signup
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        cursor = db.cursor()
        cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", 
                       (username, email, password))
        db.commit()
        return redirect(url_for('signin'))
    return render_template('signup.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        print("POST request received for sign-in.")
        username = request.form['username']
        password = request.form['password']
        
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        user = cursor.fetchone()
        
        if user:
            session['username'] = user[1]
            print("Redirecting to upload_resume")
            return redirect(url_for('upload_resume'))
        else:
            print("Invalid credentials")
            return "Invalid Credentials", 401
    print("GET request received for sign-in.")
    return render_template('signin.html')

@app.route('/upload_resume', methods=['GET', 'POST'])
def upload_resume():
    if request.method == 'POST':
        if 'resume' not in request.files:
            return "No file part"
        file = request.files['resume']

        if file.filename == '':
            return "No selected file"

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Open the file and process it
            with open(file_path, 'rb') as pdf_file:
                results = process_resume_and_match_jobs(pdf_file)

            # Only pass matched_jobs to the template
            matched_jobs = results["matched_jobs"]
            return render_template(
                'result.html',
                matched_jobs=matched_jobs
            )
    return render_template('upload_resume.html')

@app.route('/start_interview/<job_title>', methods=['GET', 'POST'])
def start_interview(job_title):
    # Generate questions for the specific job title
    questions = generate_questions_for_job(job_title)

    if request.method == 'POST':
        # Collect user answers from the form submission
        user_answers = request.form.to_dict()

        # Redirect to the evaluation route with user answers
        return render_template('evaluation_results.html', evaluation_results=user_answers)
    # Render interview form
    return render_template('interview.html', job_title=job_title, questions=questions)

@app.route('/evaluate_answers', methods=['POST'])
def evaluate_answers():
    # Collect user answers from the form
    user_answers = request.form.to_dict()

    # Get question-answer mapping
    question_answer_mapping = get_question_answer_mapping()

    # Evaluate answers for only answered questions
    evaluation_results = []
    for question, correct_answer in question_answer_mapping.items():
        user_answer = user_answers.get(question, "").strip()  # Get user's answer or default to empty string
        
        if user_answer:  # Only process answered questions
            result, score = evaluate_answer(question, user_answer, correct_answer) # Evaluate answer
            evaluation_results.append({
                "question": question,
                "user_answer": user_answer,
                "result": result,
                "score": f"{score:.2f}"
            })

    # Render evaluation results
    return render_template('evaluation_results.html', evaluation_results=evaluation_results)

@app.route('/results', methods=['POST', 'GET'])
def results():
    # Assuming you calculate or retrieve the job title from the form or database
    job_title = request.args.get('job_title')  # Or however you retrieve it
    return render_template('result.html', jobTitle=job_title)

if __name__ == '__main__':
    app.run(debug=True)
