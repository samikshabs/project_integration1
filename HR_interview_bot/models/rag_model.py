import fitz  # PyMuPDF for PDF handling 
from transformers import AutoModelForCausalLM, AutoTokenizer
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Initialize ChromaDB Client
chroma_client = chromadb.Client()
collection_name = "company_data"
collection = chroma_client.get_or_create_collection(collection_name)

# Load GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

models = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Predefined list of job titles
job_titles = [
    "Software Engineer", "Data Scientist", "Cloud Engineer", "Full Stack Developer", 
    "DevOps Engineer", "Front End Developer", "Back End Developer", "Mobile Application Developer",
    "Cybersecurity Analyst", "Database Administrator", "System Administrator", "Network Engineer",
    "IT Support Specialist", "Web Developer", "Product Manager", "Machine Learning Engineer",
    "IT Project Manager", "Business Analyst", "Technical Support Engineer", 
    "Quality Assurance Engineer", "Data Engineer", "AI Engineer", 
    "UX/UI Designer", "IT Consultant", "Solutions Architect", "IT Operations Manager", 
    "Chief Technology Officer (CTO)", "Security Engineer", "IT Auditor", "Software Architect",
    "Scrum Master", "Technical Writer", "Network Security Analyst", "Game Developer", 
    "Embedded Systems Engineer", "ERP Consultant", "Salesforce Developer", 
    "Big Data Engineer", "BI Developer", "Information Security Analyst", 
    "Robotics Engineer", "Cloud Solutions Architect", "Computer Vision Engineer", 
    "Site Reliability Engineer", "Penetration Tester", "Data Analyst", "Blockchain Developer",
    "IT Compliance Specialist", "Software Development Manager", "Virtual Reality Developer",
    "Infrastructure Engineer", "IT Operations Analyst", "Digital Marketing Specialist", 
    "Network Architect", "Help Desk Technician", "Configuration Manager", "Systems Analyst",
    "Database Developer", "IT Business Partner", "Cloud Consultant", "Virtualization Engineer",
    "E-commerce Specialist", "IT Trainer", "Technical Project Manager", "Mobile UX Designer",
    "Network Operations Center (NOC) Technician", "Release Manager", "IT Change Manager", 
    "Data Governance Analyst", "Performance Engineer", "BI Analyst", "SAP Consultant", 
    "Digital Transformation Consultant", "IT Asset Manager", "Game Designer", "Social Media Analyst"
]

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file.

    Args:
    - pdf_file: File-like object representing the uploaded PDF.

    Returns:
    - str: Extracted text from the PDF.
    """
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def format_resume_text(resume_text):
    """
    Formats resume text to make main headings clear and visually appealing.
    
    Args:
    - resume_text (str): Raw text extracted from the resume.

    Returns:
    - str: Formatted text for better readability.
    """
    lines = resume_text.split("\n")
    formatted_lines = []

    for line in lines:
        line = line.strip()
        if line.isupper() and len(line.split()) < 6:
            formatted_lines.append(f"\n\n### {line} ###\n")
        elif line:  
            formatted_lines.append(line)

    return "\n".join(formatted_lines)

def extract_skills_using_ai(resume_text):
    """
    Extracts skills from the resume using the GPT-Neo model.

    Args:
    - resume_text (str): Full text of the resume.

    Returns:
    - str: Extracted skills as a string.
    """
    prompt = f"""
    You are a highly intelligent resume parser. Your task is to extract the skills section from the following resume text. 
    Return only the text below the 'Skills' section. If the 'Skills' section is not found, return an empty string.

    Resume Text:
    {resume_text}
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=600)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

def process_resume_and_match_jobs(pdf_file):
    """
    Processes the resume and matches it with job descriptions in ChromaDB.

    Args:
    - pdf_file: File-like object representing the uploaded PDF.

    Returns:
    - dict: Dictionary containing matched jobs.
    """
    # Step 1: Extract raw text from the resume
    resume_text = extract_text_from_pdf(pdf_file)

    # Step 2: Retrieve job descriptions and metadata from ChromaDB
    results = collection.get(include=["documents", "metadatas"])
    job_descriptions = results["documents"]
    metadatas = results["metadatas"]  # Metadata contains jobTitle, jobUrl, and potentially jobCompany

    if not job_descriptions:
        return {
            "matched_jobs": [{"job_title": "No job descriptions available", "job_company": "", "similarity_score": 0}]
        }

    # Step 3: Match the resume text with job descriptions using TF-IDF and cosine similarity
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        all_documents = [resume_text] + job_descriptions
        tfidf_matrix = vectorizer.fit_transform(all_documents)

        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        top_indices = cosine_similarities.argsort()[-5:][::-1]  # Top 5 matches
        
        matched_jobs = [
            {
                "job_title": metadatas[i]["jobTitle"],
                "job_company": metadatas[i].get("jobCompany", "Unknown"),
                "similarity_score": round(cosine_similarities[i] * 100*1.5, 2)
            }
            for i in top_indices
        ]
    except ValueError as e:
        matched_jobs = [{"job_title": "Error in processing job matching", "job_company": "", "similarity_score": 0}]

    return {"matched_jobs": matched_jobs}


from sentence_transformers import SentenceTransformer, util

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

def evaluate_answer(question, answer, correct_answer):
    # Preprocess the text (case-insensitive, trim spaces)
    def preprocess_text(text):
        return " ".join(text.strip().lower().split())

    # Check for keyword presence
    def contains_keyword(user_answer, correct_answer):
        # Split into words and check if any predefined keyword is in the user's answer
        predefined_keywords = correct_answer.lower().split()  # Lowercase for case-insensitivity
        user_words = user_answer.lower().split()
        return any(keyword in user_words for keyword in predefined_keywords)

    # Preprocess all inputs
    question_cleaned = preprocess_text(question)
    user_answer_cleaned = preprocess_text(answer)
    correct_answer_cleaned = preprocess_text(correct_answer)

    # Keyword validation
    keyword_match = contains_keyword(user_answer_cleaned, correct_answer_cleaned)

    # Encode for semantic similarity
    question_embedding = model.encode(question_cleaned, convert_to_tensor=True)
    user_answer_embedding = model.encode(user_answer_cleaned, convert_to_tensor=True)
    correct_answer_embedding = model.encode(correct_answer_cleaned, convert_to_tensor=True)

    # Calculate cosine similarity between user and correct answer embeddings
    cosine_sim = util.pytorch_cos_sim(user_answer_embedding, correct_answer_embedding).item()
    score_percentage = round(cosine_sim * 100, 2)
    # Set a similarity threshold (e.g., 0.7 for semantic similarity)
    similarity_threshold = 0.55

    # Determine the result
    if keyword_match and cosine_sim >= similarity_threshold:
        result = "Correct"
    else:
        result = "Incorrect"
    
    return result, score_percentage



# Define the functions that generate questions for each role

def generate_software_engineer_questions():
    return [
        "What is a core principle of OOP?",
        "Which design pattern is commonly used?",
        "Which tool is used for version control?",
        "Which practice ensures code quality?",
        "Which type of testing verifies functionality?",
        "Which paradigm emphasizes state changes?"
    ]

def generate_data_scientist_questions():
    return [
        "Which technique removes outliers?",
        "Which plot type is used for distribution?",
        "Which algorithm is used for classification?",
        "Which metric evaluates model accuracy?",
        "Which library is used for visualization?",
        "Which technique handles class imbalance?"
    ]

def generate_cloud_engineer_questions():
    return [
        "What is your experience with cloud platforms like AWS, Azure, or Google Cloud?",
        "How do you ensure high availability and fault tolerance in the cloud?",
        "What are 3 cloud servicing models?",
        "Explain how you approach security in cloud environments.",
        "What are the various containerization and orchestration tools?",
        "How do you monitor cloud infrastructure and services?"
    ]

def generate_full_stack_developer_questions():
    return [
        "How do you ensure communication between the front-end and back-end of a full-stack application?",
        "What technologies do you use for building RESTful APIs?",
        "How would you handle user authentication and authorization in a web application?",
        "Explain the differences between SQL and NoSQL databases.",
        "How do you optimize the performance of both front-end and back-end systems?",
        "Can you walk us through your development process for a full-stack application?"
    ]

def generate_devops_engineer_questions():
    return [
        "What is your experience with continuous integration/continuous deployment (CI/CD)?",
        "How do you monitor system performance and ensure reliability?",
        "Can you explain the concept of infrastructure as code (IaC)?",
        "What tools do you use for automation and configuration management?",
        "How do you handle scaling in a cloud environment?",
        "Explain your approach to disaster recovery and business continuity."
    ]

def generate_front_end_developer_questions():
    return [
        "What is your experience with front-end frameworks like React, Angular, or Vue?",
        "How do you ensure responsive design across multiple devices?",
        "What is the difference between server-side rendering and client-side rendering?",
        "How do you optimize the performance of a front-end application?",
        "Can you explain the concept of state management in a front-end application?",
        "How do you handle cross-browser compatibility issues?"
    ]

def generate_back_end_developer_questions():
    return [
        "What is your experience with back-end technologies like Node.js, Python, or Ruby?",
        "How do you design scalable and maintainable APIs?",
        "Can you explain how you handle database migrations?",
        "What strategies do you use for error handling and logging?",
        "How do you ensure the security of your back-end systems?",
        "What is your approach to optimizing database queries for performance?"
    ]

def generate_mobile_application_developer_questions():
    return [
        "What is your experience with iOS and Android development?",
        "How do you manage app performance on different mobile devices?",
        "Can you explain the difference between native and hybrid mobile applications?",
        "What tools and frameworks do you use for mobile app testing?",
        "How do you handle offline functionality in mobile applications?",
        "What is your experience with mobile app security?"
    ]

def generate_cybersecurity_analyst_questions():
    return [
        "What is your experience with vulnerability assessments and penetration testing?",
        "How do you stay updated on the latest security threats and trends?",
        "Can you explain the difference between symmetric and asymmetric encryption?",
        "What strategies do you use to secure an organization's network?",
        "How do you approach incident response and handling a security breach?",
        "What tools do you use for network security monitoring?"
    ]

def generate_database_administrator_questions():
    return [
        "What is the primary database you work with?",
        "Which indexing technique do you prefer?",
        "What is the term for database replication across multiple servers?",
        "Which encryption method do you use for data security?",
        "What technology ensures database high availability?",
        "Which database type is more flexible: relational or non-relational?"
    ]

def generate_system_administrator_questions():
    return [
        "Which monitoring tool do you prefer?",
        "What server configuration is crucial for scalability?",
        "Which virtualization technology do you use?",
        "Which automation tool do you rely on?",
        "What is your preferred firewall technology?",
        "Which network protocol is most critical for server security?"
    ]

def generate_network_engineer_questions():
    return [
        "What is the most common network protocol you work with?",
        "Which tool do you use for network troubleshooting?",
        "What network performance tool do you use most often?",
        "What topology do you prefer for network design?",
        "Which encryption protocol do you use for securing networks?",
        "Which tool do you use for ensuring network redundancy?"
    ]

def generate_it_support_specialist_questions():
    return [
        "Which helpdesk ticketing system do you use?",
        "Which tool do you prefer for remote support?",
        "What is your preferred method for managing user access?",
        "What documentation tool do you use for system issues?",
        "Which troubleshooting methodology do you follow?",
        "What is the key to ensuring customer satisfaction?"
    ]

def generate_web_developer_questions():
    return [
        "Which HTML version do you primarily work with?",
        "Which framework do you use for front-end development?",
        "What layout technique do you use for responsive design?",
        "What JavaScript method is used for asynchronous operations?",
        "What tool do you use for web performance optimization?",
        "Which CSS preprocessor do you prefer?"
    ]

def generate_product_manager_questions():
    return [
        "Which tool do you use for gathering requirements?",
        "What method do you use to prioritize features?",
        "Which lifecycle stage is most critical for product success?",
        "Which team do you collaborate with most frequently?",
        "What project management methodology do you prefer?",
        "What metric do you use to measure product success?"
    ]

def generate_machine_learning_engineer_questions():
    return [
        "Which machine learning algorithm do you prefer?",
        "What imputation method do you use for missing data?",
        "Which feature selection technique do you use?",
        "What type of model did you use for your latest project?",
        "Which deep learning framework do you use?",
        "What metric do you use to evaluate model performance?"
    ]

def generate_it_project_manager_questions():
    return [
        "Which project management tool do you prefer?",
        "What is your strategy for managing project risks?",
        "Which agile methodology do you follow?",
        "What is the most important factor in keeping your team motivated?",
        "Which performance metric do you track in projects?",
        "Which software is most important for project documentation?"
    ]


# Business Analyst
def generate_business_analyst_questions():
    return [
        "Which method do you use to document business requirements?",
        "What tool do you prefer for gap analysis?",
        "What is your preferred technique for resolving conflicting requirements?",
        "Which process improvement methodology do you use?",
        "Which tool do you use for creating wireframes?",
        "Which business metric is most critical to aligning solutions?"
    ]

# Technical Support Engineer
def generate_technical_support_engineer_questions():
    return [
        "Which troubleshooting method do you follow?",
        "What tool do you use for handling escalated tickets?",
        "What software do you use for remote troubleshooting?",
        "What term describes explaining technical concepts to non-technical users?",
        "What method do you use to resolve recurring issues?",
        "Which resource keeps you updated on new troubleshooting techniques?"
    ]

# Quality Assurance Engineer
def generate_quality_assurance_engineer_questions():
    return [
        "Which technique do you use to design test cases?",
        "What type of testing ensures no new issues are introduced?",
        "Which prioritization method do you use for bugs?",
        "Which automation tool do you use for testing?",
        "What method do you follow when critical bugs are found late?",
        "What is the term for collaboration between developers and testers?"
    ]

# Data Engineer
def generate_data_engineer_questions():
    return [
        "What is your preferred tool for designing data pipelines?",
        "Which distributed processing system do you use?",
        "What technique do you use to ensure data integrity in ETL?",
        "Which platform do you use for real-time data processing?",
        "Which cloud platform do you use for data engineering tasks?",
        "What is your preferred method for securing sensitive data?"
    ]

# Artificial Intelligence Engineer
def generate_ai_engineer_questions():
    return [
        "Which algorithm do you prefer for model training?",
        "What technique do you use to optimize neural networks?",
        "What project demonstrates your application of AI to real-world problems?",
        "Which AI framework do you use for development?",
        "What is your approach to handling bias in AI models?",
        "What tool do you use to deploy AI models into production?"
    ]

# UX/UI Designer
def generate_ux_ui_designer_questions():
    return [
        "What method do you use for user research?",
        "Which tool do you use for wireframing?",
        "What usability improvement did you make in your last project?",
        "What strategy do you use to balance user needs with business goals?",
        "Which tool do you use for collaborating with developers?",
        "What technique do you use to gather user feedback?"
    ]

# IT Consultant
def generate_it_consultant_questions():
    return [
        "Which methodology do you use for assessing IT infrastructure?",
        "What project demonstrated your successful IT changes?",
        "What methodology do you use for aligning IT projects with business goals?",
        "Which resource keeps you updated with emerging technologies?",
        "What method do you use to manage resistance to IT changes?",
        "Which tool do you use for IT audits?"
    ]

# Solutions Architect
def generate_solutions_architect_questions():
    return [
        "Which architecture framework do you follow?",
        "What is your preferred method for designing scalable systems?",
        "Which cloud platform is your primary experience?",
        "What factor influences your decisions on cost vs. technical trade-offs?",
        "What solution did you design to solve a critical problem?",
        "Which tool do you use to validate solution requirements?"
    ]

# IT Operations Manager
def generate_it_operations_manager_questions():
    return [
        "What tool do you use to automate IT operations?",  # Answer: Ansible
        "Which metric measures system uptime?",  # Answer: Availability
        "What is the primary protocol for secure communication over a network?",  # Answer: HTTPS
        "Which standard governs IT service management processes?",  # Answer: ITIL
        "What is the most critical component in high-availability systems?",  # Answer: Redundancy
        "What method do you use for disaster recovery planning?",  # Answer: RTO (Recovery Time Objective)
    ]

# Chief Technology Officer (CTO)
def generate_cto_questions():
    return [
        "Which framework is commonly used for technology strategy alignment?",  # Answer: TOGAF
        "What is the key measure of a technology’s scalability?",  # Answer: Throughput
        "Which cloud platform is known for its serverless computing?",  # Answer: AWS Lambda
        "What is the critical phase in digital transformation?",  # Answer: Integration
        "Which technology is used for enterprise architecture modeling?",  # Answer: ArchiMate
        "What is the most common database for unstructured data?",  # Answer: MongoDB
    ]

# Security Engineer
def generate_security_engineer_questions():
    return [
        "What is the most effective encryption algorithm for data security?",  # Answer: AES
        "What protocol is widely used for secure email communication?",  # Answer: PGP
        "What term refers to a network attack that floods a target with traffic?",  # Answer: DDoS
        "Which security framework is used to manage risk in an organization?",  # Answer: NIST
        "What technology is used for endpoint protection against malware?",  # Answer: EDR
        "What is the key standard for securing wireless networks?",  # Answer: WPA3
    ]

# IT Auditor
def generate_it_auditor_questions():
    return [
        "What is the most common framework used for IT governance?",  # Answer: COBIT
        "What standard is critical for evaluating information security?",  # Answer: ISO 27001
        "What is the first step in conducting a vulnerability assessment?",  # Answer: Scanning
        "Which tool is used to monitor system security events?",  # Answer: SIEM
        "What is the main objective of an IT audit?",  # Answer: Compliance
        "What is the key document produced during an IT audit?",  # Answer: Report
    ]

# Software Architect
def generate_software_architect_questions():
    return [
        "What architectural style is used for microservices?",  # Answer: SOA (Service-Oriented Architecture)
        "What is the most widely used pattern for distributed systems?",  # Answer: Event-driven
        "Which protocol is commonly used for real-time web applications?",  # Answer: WebSocket
        "What design pattern is best for decoupling object creation?",  # Answer: Factory
        "What database model is ideal for hierarchical data?",  # Answer: NoSQL
        "What technique is used to ensure fault tolerance in a distributed system?",  # Answer: Replication
    ]

# Scrum Master
def generate_scrum_master_questions():
    return [
        "What agile methodology is used for managing iterative project work?",  # Answer: Scrum
        "What is the typical timebox for a sprint?",  # Answer: Two weeks
        "Which metric is used to measure team performance in Scrum?",  # Answer: Velocity
        "What tool is commonly used for agile project tracking?",  # Answer: Jira
        "What is the primary role of the Scrum Master?",  # Answer: Facilitation
        "What is the Scrum event where the team plans work for the next sprint?",  # Answer: Sprint Planning
    ]

# Technical Writer
def generate_technical_writer_questions():
    return [
        "What markup language is used to write documentation for technical content?",  # Answer: Markdown
        "What process do you use to organize large sets of documentation?",  # Answer: Structuring
        "Which tool is preferred for creating and editing technical content?",  # Answer: Confluence
        "What document type is commonly used for API documentation?",  # Answer: Swagger
        "Which standard defines the style and formatting of technical documents?",  # Answer: Chicago
        "What writing technique is essential for simplifying complex information?",  # Answer: Clarity
    ]

# Network Security Analyst
def generate_network_security_analyst_questions():
    return [
        "Which protocol is used for securing email communication?",  # Answer: SMTP
        "What tool is commonly used for network vulnerability scanning?",  # Answer: Nessus
        "What technology provides encrypted tunnels for secure communication?",  # Answer: VPN
        "Which algorithm is commonly used for public-key encryption?",  # Answer: RSA
        "What is the standard for network security monitoring?",  # Answer: SNMP
        "Which attack attempts to overwhelm a network with traffic?",  # Answer: Flooding
    ]

# Game Developer
def generate_game_developer_questions():
    return [
        "Which game engine is widely used for 3D game development?",  # Answer: Unity
        "What technology is used for physics simulation in games?",  # Answer: Havok
        "Which language is commonly used for scripting in Unreal Engine?",  # Answer: Blueprints
        "What technique is used to reduce the file size of game textures?",  # Answer: Compression
        "What method is used to synchronize multiplayer game states?",  # Answer: RPC (Remote Procedure Call)
        "What is the most common data structure for representing a game world?",  # Answer: Graph
    ]

# Embedded Systems Engineer
def generate_embedded_systems_engineer_questions():
    return [
        "Which protocol is commonly used for communication in embedded systems?",  # Answer: SPI
        "What is the primary language used for embedded software development?",  # Answer: C
        "What tool is commonly used for debugging embedded systems?",  # Answer: JTAG
        "What is the most common architecture for embedded systems?",  # Answer: ARM
        "Which operating system is used for real-time embedded systems?",  # Answer: RTOS
        "What is the primary concern for embedded systems in battery-powered devices?",  # Answer: Power
    ]


# ERP Consultant
def generate_erp_consultant_questions():
    return [
        "What ERP platform is widely used for large enterprises?",  # Answer: SAP
        "Which methodology is often used for ERP system implementation?",  # Answer: ASAP
        "What tool do you use for ERP data migration?",  # Answer: LSMW
        "What is the most important factor in ERP customization?",  # Answer: Flexibility
        "Which programming language is used for customizing SAP ERP?",  # Answer: ABAP
        "What is the standard for ERP system integration?",  # Answer: Middleware
    ]

# Salesforce Developer
def generate_salesforce_developer_questions():
    return [
        "What is the primary language used for custom development in Salesforce?",  # Answer: Apex
        "Which framework is commonly used for building UI in Salesforce?",  # Answer: Visualforce
        "What is the most commonly used tool for Salesforce data integration?",  # Answer: MuleSoft
        "Which method is used to query data in Salesforce?",  # Answer: SOQL
        "What type of cloud is Salesforce primarily known for?",  # Answer: CRM
        "What Salesforce product is used for customer service management?",  # Answer: ServiceCloud
    ]

# Big Data Engineer
def generate_big_data_engineer_questions():
    return [
        "What framework is most commonly used for distributed data processing?",  # Answer: Hadoop
        "Which programming language is typically used for Spark applications?",  # Answer: Scala
        "What is the default storage format in Hadoop?",  # Answer: HDFS
        "Which tool is most commonly used for real-time big data processing?",  # Answer: Kafka
        "What is the main advantage of using NoSQL databases for big data?",  # Answer: Scalability
        "Which tool is used for batch processing in big data systems?",  # Answer: Hive
    ]

# BI Developer
def generate_bi_developer_questions():
    return [
        "Which BI tool is commonly used for data visualization?",  # Answer: Tableau
        "What query language is used in Power BI for data modeling?",  # Answer: DAX
        "What is the most common format for storing BI data?",  # Answer: OLAP
        "Which database system is typically used for BI reporting?",  # Answer: SQL Server
        "What is the key principle of the ETL process?",  # Answer: Extraction
        "Which BI tool is known for its integration with Excel?",  # Answer: Power BI
    ]

# Information Security Analyst
def generate_information_security_analyst_questions():
    return [
        "What tool is commonly used for vulnerability scanning?",  # Answer: Nessus
        "What encryption algorithm is widely used for securing web traffic?",  # Answer: AES
        "Which standard is used for information security management?",  # Answer: ISO 27001
        "What is the most common method for network intrusion detection?",  # Answer: IDS
        "Which attack method involves overwhelming a system with requests?",  # Answer: DDoS
        "What is the best practice for securing passwords?",  # Answer: Hashing
    ]

# Robotics Engineer
def generate_robotics_engineer_questions():
    return [
        "What programming language is commonly used for robotic systems?",  # Answer: C++
        "Which software framework is used for controlling robots?",  # Answer: ROS
        "What sensor is most commonly used for measuring distance in robots?",  # Answer: LIDAR
        "What is the key factor for real-time performance in robotics?",  # Answer: Latency
        "Which algorithm is used for robot path planning?",  # Answer: A*
        "What is the primary challenge in autonomous robotics?",  # Answer: Localization
    ]

# Cloud Solutions Architect
def generate_cloud_solutions_architect_questions():
    return [
        "What is the most widely used cloud provider?",  # Answer: AWS
        "Which cloud service model provides the highest level of control?",  # Answer: IaaS
        "What is the most popular cloud storage solution?",  # Answer: S3
        "Which protocol is used for secure communication in cloud environments?",  # Answer: HTTPS
        "What technology enables automatic scaling of cloud applications?",  # Answer: Autoscaling
        "What is the main tool for managing cloud infrastructure as code?",  # Answer: Terraform
    ]

# Computer Vision Engineer
def generate_computer_vision_engineer_questions():
    return [
        "Which framework is commonly used for deep learning in computer vision?",  # Answer: TensorFlow
        "What technique is used for object detection in images?",  # Answer: CNN
        "What is the most common metric for evaluating object detection performance?",  # Answer: mAP
        "What algorithm is used for facial recognition?",  # Answer: LBP
        "Which tool is widely used for image processing in computer vision?",  # Answer: OpenCV
        "What technique is used to handle lighting variations in images?",  # Answer: Histogram
    ]

# Site Reliability Engineer
def generate_site_reliability_engineer_questions():
    return [
        "What is the main tool used for infrastructure automation?",  # Answer: Ansible
        "Which service is used for monitoring distributed systems?",  # Answer: Prometheus
        "What is the term for managing server configurations in code?",  # Answer: IaC
        "Which metric is critical for evaluating system reliability?",  # Answer: SLA
        "What is the most common tool for continuous integration in DevOps?",  # Answer: Jenkins
        "Which tool is used for container orchestration?",  # Answer: Kubernetes
    ]

# Penetration Tester
def generate_penetration_tester_questions():
    return [
        "What is the primary tool for scanning network vulnerabilities?",  # Answer: Nmap
        "Which attack method is used for gaining unauthorized access to a system?",  # Answer: Exploitation
        "What is the best tool for password cracking?",  # Answer: John
        "Which protocol is most often targeted in network penetration testing?",  # Answer: SMB
        "What is the most commonly used technique for web application testing?",  # Answer: SQL Injection
        "What type of penetration test is conducted with permission from the system owner?",  # Answer: Ethical
    ]

# Data Analyst
def generate_data_analyst_questions():
    return [
        "Which tool is most commonly used for data analysis in businesses?",  # Answer: Excel
        "What is the most common language for data manipulation?",  # Answer: Python
        "What is the most important aspect of data cleaning?",  # Answer: Consistency
        "Which statistical test is commonly used for hypothesis testing?",  # Answer: T-test
        "What is the most common format for structured data?",  # Answer: CSV
        "What is the most widely used database for data analysis?",  # Answer: SQL
    ]

# Blockchain Developer
def generate_blockchain_developer_questions():
    return [
        "Which blockchain platform is known for its smart contract functionality?",  # Answer: Ethereum
        "What language is most commonly used for writing Ethereum smart contracts?",  # Answer: Solidity
        "Which consensus algorithm does Bitcoin use?",  # Answer: Proof-of-Work
        "What tool is commonly used for testing Ethereum contracts?",  # Answer: Truffle
        "What is the most popular blockchain for enterprise use?",  # Answer: Hyperledger
        "Which type of blockchain ensures complete transparency and immutability?",  # Answer: Public
    ]

# IT Compliance Specialist
def generate_it_compliance_specialist_questions():
    return [
        "Which regulation is focused on data protection and privacy in the EU?",  # Answer: GDPR
        "What framework is most commonly used for IT compliance in security?",  # Answer: ISO 27001
        "Which standard is widely adopted for IT service management?",  # Answer: ITIL
        "What is the most common audit framework used in IT?",  # Answer: SOC 2
        "Which tool is commonly used for IT compliance monitoring?",  # Answer: GRC
        "What is the most critical component in maintaining IT compliance?",  # Answer: Documentation
    ]

# Software Development Manager
def generate_software_development_manager_questions():
    return [
        "Which methodology emphasizes iterative development and collaboration?",  # Answer: Agile
        "What tool is commonly used for version control in software development?",  # Answer: Git
        "Which programming language is known for its use in enterprise software?",  # Answer: Java
        "What is the main concept behind DevOps?",  # Answer: Automation
        "What is the most common model for project management in software development?",  # Answer: Scrum
        "What is the primary responsibility of a software development manager?",  # Answer: Leadership
    ]

# Virtual Reality Developer
def generate_virtual_reality_developer_questions():
    return [
        "Which game engine is commonly used for VR development?",  # Answer: Unity
        "What is the most common VR hardware used for immersive experiences?",  # Answer: Oculus
        "Which technology is commonly integrated with VR for real-time interactivity?",  # Answer: AI
        "What is the most important consideration in designing VR user interfaces?",  # Answer: Comfort
        "What is the primary challenge in VR content creation?",  # Answer: Motion sickness
        "Which programming language is most commonly used for VR development?",  # Answer: C#
    ]

# Site Reliability Engineer
def generate_site_reliability_engineer_questions():
    return [
        "Which tool is commonly used for monitoring system performance?",  # Answer: Prometheus
        "What methodology focuses on the reliability and performance of services?",  # Answer: SRE
        "What is the most common metric for measuring service availability?",  # Answer: Uptime
        "What is the most commonly used technique for automating infrastructure?",  # Answer: Terraform
        "What tool is used for container orchestration in cloud environments?",  # Answer: Kubernetes
        "What is the primary focus of a site reliability engineer?",  # Answer: Reliability
    ]

# Infrastructure Engineer
def generate_infrastructure_engineer_questions():
    return [
        "What cloud platform is most commonly used for building infrastructure?",  # Answer: AWS
        "What is the most common virtualization technology for managing servers?",  # Answer: VMware
        "Which network protocol is most commonly used for securing communications?",  # Answer: HTTPS
        "What is the most important factor when designing scalable infrastructure?",  # Answer: Load balancing
        "Which system is used for infrastructure automation?",  # Answer: Ansible
        "What type of network is used to connect cloud-based resources?",  # Answer: VPC
    ]

# IT Operations Analyst
def generate_it_operations_analyst_questions():
    return [
        "What is the most common framework for IT service management?",  # Answer: ITIL
        "Which tool is widely used for incident management?",  # Answer: ServiceNow
        "What is the key goal of IT operations management?",  # Answer: Efficiency
        "What process is used to restore normal service after an incident?",  # Answer: Incident resolution
        "What is the primary goal of a monitoring tool in IT operations?",  # Answer: Uptime
        "What is the most important aspect of a post-incident review?",  # Answer: Analysis
    ]

# Digital Marketing Specialist
def generate_digital_marketing_specialist_questions():
    return [
        "What is the most common metric for measuring website traffic?",  # Answer: Visits
        "Which tool is commonly used for SEO analysis?",  # Answer: Google Analytics
        "What is the most common social media platform for B2B marketing?",  # Answer: LinkedIn
        "Which advertising model is based on pay-per-click?",  # Answer: PPC
        "What is the primary goal of content marketing?",  # Answer: Engagement
        "What is the most common format for digital marketing ads?",  # Answer: Display
    ]

# Network Architect
def generate_network_architect_questions():
    return [
        "Which protocol is most commonly used for routing in networks?",  # Answer: BGP
        "What technology is used to virtualize network resources?",  # Answer: SDN
        "Which type of network is typically used to interconnect data centers?",  # Answer: WAN
        "What is the most important factor when designing secure network architectures?",  # Answer: Firewalls
        "Which tool is commonly used for network performance monitoring?",  # Answer: Wireshark
        "What is the primary goal of a network architect?",  # Answer: Scalability
    ]

# Help Desk Technician
def generate_help_desk_technician_questions():
    return [
        "Which tool is widely used for ticketing and issue tracking?",  # Answer: Jira
        "What is the first step in troubleshooting an IT issue?",  # Answer: Diagnosis
        "Which operating system is most commonly used in enterprise IT environments?",  # Answer: Windows
        "What method is commonly used to solve recurring IT issues?",  # Answer: Documentation
        "What is the primary role of a help desk technician?",  # Answer: Support
        "Which service is typically used for remote IT support?",  # Answer: TeamViewer
    ]

# Configuration Manager
def generate_configuration_manager_questions():
    return [
        "What is the most commonly used tool for configuration management?",  # Answer: Ansible
        "Which version control system is most widely used in IT?",  # Answer: Git
        "What is the primary goal of configuration management?",  # Answer: Consistency
        "Which framework is most commonly used for change management?",  # Answer: ITIL
        "What is the most critical factor in maintaining configuration standards?",  # Answer: Auditing
        "What is the primary challenge in configuration management?",  # Answer: Complexity
    ]

# Systems Analyst
def generate_systems_analyst_questions():
    return [
        "Which tool is commonly used for process modeling?",  # Answer: BPMN
        "What method is most commonly used for requirements gathering?",  # Answer: Interviews
        "What is the primary focus of a systems analyst?",  # Answer: Optimization
        "Which software is commonly used for systems integration testing?",  # Answer: Selenium
        "What is the most important factor when analyzing system performance?",  # Answer: Efficiency
        "What is the most important consideration when implementing a new system?",  # Answer: Scalability
    ]

# Database Developer
def generate_database_developer_questions():
    return [
        "Which SQL command is used to retrieve data from a database?",  # Answer: SELECT
        "What is the most important factor in database optimization?",  # Answer: Indexing
        "Which programming language is most commonly used for database procedures?",  # Answer: PL/SQL
        "What is the most common type of database used in enterprise applications?",  # Answer: Relational
        "Which tool is commonly used for database management?",  # Answer: MySQL
        "What is the key principle behind database normalization?",  # Answer: Minimization
    ]

# IT Business Partner
def generate_it_business_partner_questions():
    return [
        "Which framework is commonly used to align IT and business strategies?",  # Answer: COBIT
        "What is the most common metric for measuring IT performance?",  # Answer: ROI
        "Which process is most important for managing IT resources?",  # Answer: Budgeting
        "What is the primary role of an IT business partner?",  # Answer: Alignment
        "What is the most important factor when identifying digital transformation opportunities?",  # Answer: Innovation
        "Which tool is commonly used for IT project tracking?",  # Answer: Jira
    ]

# Cloud Consultant
def generate_cloud_consultant_questions():
    return [
        "Which cloud platform is most widely used for enterprise applications?",  # Answer: AWS
        "What is the primary advantage of cloud computing?",  # Answer: Scalability
        "Which service is commonly used for serverless computing?",  # Answer: Lambda
        "What is the most important factor when designing a cloud architecture?",  # Answer: Flexibility
        "Which tool is commonly used for managing cloud costs?",  # Answer: CloudWatch
        "What is the most important aspect of cloud security?",  # Answer: Encryption
    ]

# Virtualization Engineer
def generate_virtualization_engineer_questions():
    return [
        "What is the most widely used virtualization platform?",  # Answer: VMware
        "Which technology is often used to containerize applications?",  # Answer: Docker
        "What is the primary benefit of virtualization?",  # Answer: Efficiency
        "Which type of virtualization is commonly used for server management?",  # Answer: Hypervisor
        "What is the most important factor when optimizing a virtualized environment?",  # Answer: Resource allocation
        "Which tool is most commonly used for managing virtual machines?",  # Answer: vSphere
    ]

# E-commerce Specialist
def generate_e_commerce_specialist_questions():
    return [
        "Which e-commerce platform is most widely used for online stores?",  # Answer: Shopify
        "What is the most important factor in increasing online sales?",  # Answer: Conversion
        "Which metric is most commonly used to measure e-commerce success?",  # Answer: Revenue
        "What is the most common payment gateway used in e-commerce?",  # Answer: PayPal
        "What is the key strategy for optimizing e-commerce websites?",  # Answer: UX/UI
        "Which tool is commonly used to track e-commerce metrics?",  # Answer: Google Analytics
    ]

# IT Trainer
def generate_it_trainer_questions():
    return [
        "Which platform is commonly used for online IT training?",  # Answer: Udemy
        "What is the most important factor when designing training programs?",  # Answer: Engagement
        "What tool is widely used for virtual classrooms?",  # Answer: Zoom
        "What is the most common method for assessing training effectiveness?",  # Answer: Feedback
        "Which technique is most effective for teaching complex IT concepts?",  # Answer: Hands-on
        "What is the most important consideration when customizing training materials?",  # Answer: Audience
    ]

# Technical Project Manager
def generate_technical_project_manager_questions():
    return [
        "Which methodology is most commonly used for managing technical projects?",  # Answer: Agile
        "What is the primary responsibility of a technical project manager?",  # Answer: Delivery
        "Which tool is most widely used for project tracking?",  # Answer: Jira
        "What is the most common method for managing project risks?",  # Answer: Mitigation
        "Which project management process is most crucial for scope management?",  # Answer: Change control
        "What is the most important aspect when managing a project budget?",  # Answer: Accuracy
    ]

# Mobile UX Designer
def generate_mobile_ux_designer_questions():
    return [
        "Which tool is most commonly used for mobile wireframing?",  # Answer: Figma
        "What is the most important aspect of mobile UX design?",  # Answer: Simplicity
        "Which platform is most commonly used for mobile app testing?",  # Answer: TestFlight
        "What is the primary challenge in designing mobile interfaces?",  # Answer: Screen size
        "Which design principle is crucial for mobile UX?",  # Answer: Responsiveness
        "What is the most important factor in mobile app usability?",  # Answer: Navigation
    ]

# Network Operations Center (NOC) Technician
def generate_noc_technician_questions():
    return [
        "What is the most commonly used tool for network monitoring?",  # Answer: Nagios
        "What is the first step in troubleshooting network issues?",  # Answer: Diagnosis
        "Which protocol is primarily used for remote management of network devices?",  # Answer: SNMP
        "What is the key principle behind network troubleshooting?",  # Answer: Isolation
        "What is the most important factor when prioritizing network incidents?",  # Answer: Severity
        "What is the most common network maintenance task?",  # Answer: Monitoring
    ]

# Release Manager
def generate_release_manager_questions():
    return [
        "What tool is most commonly used for version control in software development?",  # Answer: Git
        "What is the primary goal of a release manager?",  # Answer: Coordination
        "What process is used to ensure a smooth deployment?",  # Answer: Automation
        "Which framework is widely used for continuous integration?",  # Answer: Jenkins
        "What is the most critical factor in a successful release?",  # Answer: Communication
        "What tool is used for tracking post-release feedback?",  # Answer: Jira
    ]

# IT Change Manager
def generate_it_change_manager_questions():
    return [
        "What framework is commonly used for IT change management?",  # Answer: ITIL
        "What tool is most commonly used for change request tracking?",  # Answer: ServiceNow
        "What is the primary objective of change management?",  # Answer: Control
        "Which process is used to assess the impact of changes?",  # Answer: Risk assessment
        "What is the most important factor in managing IT changes?",  # Answer: Communication
        "What is the most common type of IT change?",  # Answer: Updates
    ]

# Data Governance Analyst
def generate_data_governance_analyst_questions():
    return [
        "What is the most important regulation for data protection?",  # Answer: GDPR
        "What is the most commonly used tool for data governance?",  # Answer: Collibra
        "What is the primary goal of data governance?",  # Answer: Integrity
        "Which data management practice is crucial for ensuring compliance?",  # Answer: Auditing
        "What is the key focus when establishing data governance policies?",  # Answer: Consistency
        "What is the first step in managing data lineage?",  # Answer: Mapping
    ]

# Performance Engineer
def generate_performance_engineer_questions():
    return [
        "Which tool is most commonly used for load testing?",  # Answer: JMeter
        "What metric is most critical for evaluating system performance?",  # Answer: Latency
        "What is the most important factor when addressing performance bottlenecks?",  # Answer: Profiling
        "Which technology is commonly used for performance monitoring?",  # Answer: Prometheus
        "What is the key metric when evaluating application performance under heavy load?",  # Answer: Throughput
        "What is the primary focus of performance tuning?",  # Answer: Optimization
    ]

# BI Analyst
def generate_bi_analyst_questions():
    return [
        "Which BI tool is most widely used for data visualization?",  # Answer: Tableau
        "What metric is most crucial for evaluating BI effectiveness?",  # Answer: Accuracy
        "Which process is essential when validating data accuracy?",  # Answer: Cleansing
        "What type of analysis is commonly used to interpret BI data?",  # Answer: Trend
        "What is the most important factor in aligning BI solutions with business goals?",  # Answer: Relevance
        "What is the most common method for gathering business requirements?",  # Answer: Interviews
    ]

# SAP Consultant
def generate_sap_consultant_questions():
    return [
        "Which SAP module is most commonly implemented in finance?",  # Answer: FICO
        "What tool is widely used for SAP system troubleshooting?",  # Answer: Solution Manager
        "What is the key benefit of using SAP S/4HANA?",  # Answer: Speed
        "Which process is most critical for successful SAP implementation?",  # Answer: Configuration
        "What is the most important factor in ensuring SAP system integration?",  # Answer: Compatibility
        "Which type of SAP system is primarily used for data migration?",  # Answer: ETL
    ]

# Digital Transformation Consultant
def generate_digital_transformation_consultant_questions():
    return [
        "Which technology is most commonly used in digital transformation?",  # Answer: Cloud
        "What is the most critical aspect when managing digital transformation?",  # Answer: Change
        "Which framework is widely used to assess digital maturity?",  # Answer: DMM
        "What is the most important factor in ensuring alignment with business goals?",  # Answer: Strategy
        "Which KPI is most commonly used to measure digital transformation success?",  # Answer: ROI
        "What method is commonly used to manage resistance to digital change?",  # Answer: Communication
    ]

# IT Asset Manager
def generate_it_asset_manager_questions():
    return [
        "What tool is most commonly used for IT asset tracking?",  # Answer: ServiceNow
        "What is the primary goal of IT asset management?",  # Answer: Optimization
        "Which process is most critical for managing software licenses?",  # Answer: Auditing
        "What is the most important factor in managing asset lifecycles?",  # Answer: Planning
        "Which action is essential when disposing of IT assets?",  # Answer: Sanitization
        "What is the primary focus when improving IT asset management efficiency?",  # Answer: Automation
    ]

# Game Designer
def generate_game_designer_questions():
    return [
        "Which game engine is most commonly used for 3D game design?",  # Answer: Unreal
        "What is the most important aspect of gameplay design?",  # Answer: Engagement
        "Which tool is most commonly used for prototyping game concepts?",  # Answer: Unity
        "What is the most critical factor in balancing game design?",  # Answer: Playability
        "Which programming language is widely used in game development?",  # Answer: C#
        "What is the most important element in creating a game narrative?",  # Answer: Storytelling
    ]

# Social Media Analyst
def generate_social_media_analyst_questions():
    return [
        "What tool is most commonly used for social media performance analysis?",  # Answer: Hootsuite
        "What metric is most important for measuring social media ROI?",  # Answer: Engagement
        "Which platform is most critical for tracking brand sentiment?",  # Answer: Twitter
        "What method is widely used for audience segmentation?",  # Answer: Demographics
        "Which type of analysis is crucial for improving social media engagement?",  # Answer: A/B testing
        "What is the key factor in analyzing competitor performance on social media?",  # Answer: Benchmarking
    ]

def generate_questions_for_job(job_title):

    questions = {
        "Software Engineer": generate_software_engineer_questions(),
        "Data Scientist": generate_data_scientist_questions(),
        "Cloud Engineer": generate_cloud_engineer_questions(),
        "Full Stack Developer": generate_full_stack_developer_questions(),
        "DevOps Engineer": generate_devops_engineer_questions(),
        "Front End Developer": generate_front_end_developer_questions(),
        "Back End Developer": generate_back_end_developer_questions(),
        "Mobile Application Developer": generate_mobile_application_developer_questions(),
        "Cybersecurity Analyst": generate_cybersecurity_analyst_questions(),
        "Database Administrator": generate_database_administrator_questions(),
        "System Administrator": generate_system_administrator_questions(),
        "Network Engineer": generate_network_engineer_questions(),
        "IT Support Specialist": generate_it_support_specialist_questions(),
        "Web Developer": generate_web_developer_questions(),
        "Product Manager": generate_product_manager_questions(),
        "Machine Learning Engineer": generate_machine_learning_engineer_questions(),
        "IT Project Manager": generate_it_project_manager_questions(),
        "Business Analyst": generate_business_analyst_questions(),
        "Technical Support Engineer": generate_technical_support_engineer_questions(),
        "Quality Assurance Engineer": generate_quality_assurance_engineer_questions(),
        "Data Engineer": generate_data_engineer_questions(),
        "AI Engineer": generate_ai_engineer_questions(),
        "UI/UX Designer": generate_ux_ui_designer_questions(),
        "IT Consultant": generate_it_consultant_questions(),
        "Solutions Architect": generate_solutions_architect_questions(),
        "IT Operations Manager": generate_it_operations_manager_questions(),
        "Chief Technology Officer": generate_cto_questions(),
        "Security Engineer": generate_security_engineer_questions(),
        "IT Auditor": generate_it_auditor_questions(),
        "Software Architect": generate_software_architect_questions(),
        "Scrum Master": generate_scrum_master_questions(),
        "Technical Writer": generate_technical_writer_questions(),
        "Network Security Analyst": generate_network_security_analyst_questions(),
        "Game Developer": generate_game_developer_questions(),
        "Embedded Systems Engineer": generate_embedded_systems_engineer_questions(),
        "ERP Consultant": generate_erp_consultant_questions(),
        "Salesforce Developer": generate_salesforce_developer_questions(),
        "Big Data Engineer": generate_big_data_engineer_questions(),
        "BI Developer": generate_bi_developer_questions(),
        "Information Security Analyst": generate_information_security_analyst_questions(),
        "Robotics Engineer": generate_robotics_engineer_questions(),
        "Cloud Solutions Architect": generate_cloud_solutions_architect_questions(),
        "Computer Vision Engineer": generate_computer_vision_engineer_questions(),
        "Site Reliability Engineer": generate_site_reliability_engineer_questions(),
        "Penetration Tester": generate_penetration_tester_questions(),
        "Data Analyst": generate_data_analyst_questions(),
        "Blockchain Developer": generate_blockchain_developer_questions(),
        "IT Compliance Specialist": generate_it_compliance_specialist_questions(),
        "Software Development Manager": generate_software_development_manager_questions(),
        "Virtual Reality Developer": generate_virtual_reality_developer_questions(),
        "Infrastructure Engineer": generate_infrastructure_engineer_questions(),
        "IT Operations Analyst": generate_it_operations_analyst_questions(),
        "Digital Marketing Specialist": generate_digital_marketing_specialist_questions(),
        "Network Architect": generate_network_architect_questions(),
        "Help Desk Technician": generate_help_desk_technician_questions(),
        "Configuration Manager": generate_configuration_manager_questions(),
        "Systems Analyst": generate_systems_analyst_questions(),
        "Database Developer": generate_database_developer_questions(),
        "IT Business Partner": generate_it_business_partner_questions(),
        "Cloud Consultant": generate_cloud_consultant_questions(),
        "Virtualization Engineer": generate_virtualization_engineer_questions(),
        "E-commerce Specialist": generate_e_commerce_specialist_questions(),
        "IT Trainer": generate_it_trainer_questions(),
        "Technical Project Manager": generate_technical_project_manager_questions(),
        "Mobile UX Designer": generate_mobile_ux_designer_questions(),
        "Network Operations Center (NOC) Technician": generate_noc_technician_questions(),
        "Release Manager": generate_release_manager_questions(),
        "IT Change Manager": generate_it_change_manager_questions(),
        "Data Governance Analyst": generate_data_governance_analyst_questions(),
        "Performance Engineer": generate_performance_engineer_questions(),
        "BI Analyst": generate_bi_analyst_questions(),
        "SAP Consultant": generate_sap_consultant_questions(),
        "Digital Transformation Consultant": generate_digital_transformation_consultant_questions(),
        "IT Asset Manager": generate_it_asset_manager_questions(),
        "Game Designer": generate_game_designer_questions(),
        "Social Media Analyst": generate_social_media_analyst_questions()
    }
     # Normalize job title to avoid case or whitespace issues
    job_title = job_title.strip()  # Remove leading/trailing spaces
    if job_title in questions:
        # Debugging log
        print(f"Found job title: {job_title}")
        return questions[job_title]
    else:
        # Debugging log
        print(f"Job title not found: {job_title}. Returning default questions.")
        return [
            f"What relevant experience do you have for the role of {job_title}?",
            f"What technologies have you worked with that are essential for {job_title}?",
            f"Describe a challenge you faced in a similar role to {job_title} and how you resolved it.",
            f"How would you approach a project in the position of {job_title}?"
        ]

def get_question_answer_mapping():
    return {
        # Software Engineer Questions and Answers
        "What is a core principle of OOP?": "Inheritance",
        "Which design pattern is commonly used?": "Singleton",
        "Which tool is used for version control?": "Git",
        "Which practice ensures code quality?": "Refactoring",
        "Which type of testing verifies functionality?": "Unit",
        "Which paradigm emphasizes state changes?": "Imperative",
        
        # Data Scientist Questions and Answers
        "Which technique removes outliers?": "Imputation",
        "Which plot type is used for distribution?": "Histogram",
        "Which algorithm is used for classification?": "Logistic",
        "Which metric evaluates model accuracy?": "Precision",
        "Which library is used for visualization?": "Matplotlib",
        "Which technique handles class imbalance?": "Oversampling",

        # Cloud Engineer Questions and Answers
        ("What is your experience with cloud platforms like AWS, Azure, or Google Cloud?"): "AWS, Azure, Google Cloud",
        ("How do you ensure high availability and fault tolerance in the cloud?"): "Load balancing",
        ("What are 3 cloud servicing models?"): "IaaS, PaaS, SaaS",
        ("Explain how you approach security in cloud environments."): "Encryption",
        ("What are the various containerization and orchestration tools?"): "Docker, Kubernetes",
        ("How do you monitor cloud infrastructure and services?"): "CloudWatch",

        # Full Stack Developer Questions and Answers
        ("How do you ensure communication between the front-end and back-end of a full-stack application?"): "APIs",
        ("What technologies do you use for building RESTful APIs?"): "Node.js, Express",
        ("How would you handle user authentication and authorization in a web application?"): "JWT",
        ("Explain the differences between SQL and NoSQL databases."): "SQL, NoSQL",
        ("How do you optimize the performance of both front-end and back-end systems?"): "Caching",
        ("Can you walk us through your development process for a full-stack application?"): "Agile",

        # DevOps Engineer Questions and Answers
        ("What is your experience with continuous integration/continuous deployment (CI/CD)?"): "Jenkins, GitLab CI",
        ("How do you monitor system performance and ensure reliability?"): "Nagios, Prometheus",
        ("Can you explain the concept of infrastructure as code (IaC)?"): "Terraform, Ansible",
        ("What tools do you use for automation and configuration management?"): "Chef, Puppet",
        ("How do you handle scaling in a cloud environment?"): "Auto-scaling",
        ("Explain your approach to disaster recovery and business continuity."): "Backup",

        # Front-End Developer Questions and Answers
        ("What is your experience with front-end frameworks like React, Angular, or Vue?"): "React, Angular, Vue",
        ("How do you ensure responsive design across multiple devices?"): "CSS, Media queries",
        ("What is the difference between server-side rendering and client-side rendering?"): "SSR, CSR",
        ("How do you optimize the performance of a front-end application?"): "Lazy loading",
        ("Can you explain the concept of state management in a front-end application?"): "Redux",
        ("How do you handle cross-browser compatibility issues?"): "Polyfills",

        # Back-End Developer Questions and Answers
        ("What is your experience with back-end technologies like Node.js, Python, or Ruby?"): "Node.js, Python, Ruby",
        ("How do you design scalable and maintainable APIs?"): "REST, GraphQL",
        ("Can you explain how you handle database migrations?"): "Liquibase",
        ("What strategies do you use for error handling and logging?"): "Try-catch, Logs",
        ("How do you ensure the security of your back-end systems?"): "OAuth, JWT",
        ("What is your approach to optimizing database queries for performance?"): "Indexing",

        # Mobile Application Developer Questions and Answers
        ("What is your experience with iOS and Android development?"): "Swift, Kotlin",
        ("How do you manage app performance on different mobile devices?"): "Profiling",
        ("Can you explain the difference between native and hybrid mobile applications?"): "Native, Hybrid",
        ("What tools and frameworks do you use for mobile app testing?"): "Appium, XCTest",
        ("How do you handle offline functionality in mobile applications?"): "SQLite",
        ("What is your experience with mobile app security?"): "Keychain",

        # Cybersecurity Analyst Questions and Answers
        ("What is your experience with vulnerability assessments and penetration testing?"): "Nessus, Metasploit",
        ("How do you stay updated on the latest security threats and trends?"): "News, Blogs",
        ("Can you explain the difference between symmetric and asymmetric encryption?"): "AES, RSA",
        ("What strategies do you use to secure an organization's network?"): "Firewalls",
        ("How do you approach incident response and handling a security breach?"): "Containment",
        ("What tools do you use for network security monitoring?"): "Wireshark",

        # Database Administrator Questions and Answers
        ("What is the primary database you work with?"): "MySQL",
        ("Which indexing technique do you prefer?"): "B-tree",
        ("What is the term for database replication across multiple servers?"): "Replication",
        ("Which encryption method do you use for data security?"): "AES",
        ("What technology ensures database high availability?"): "Clustering",
        ("Which database type is more flexible: relational or non-relational?"): "Non-relational",

        # System Administrator Questions and Answers
        ("Which monitoring tool do you prefer?"): "Nagios",
        ("What server configuration is crucial for scalability?"): "Load balancing",
        ("Which virtualization technology do you use?"): "VMware",
        ("Which automation tool do you rely on?"): "Ansible",
        ("What is your preferred firewall technology?"): "pfSense",
        ("Which network protocol is most critical for server security?"): "SSH",

        # Network Engineer Questions and Answers
        ("What is the most common network protocol you work with?"): "TCP/IP",
        ("Which tool do you use for network troubleshooting?"): "Ping",
        ("What network performance tool do you use most often?"): "iperf",
        ("What topology do you prefer for network design?"): "Star",
        ("Which encryption protocol do you use for securing networks?"): "IPSec",
        ("Which tool do you use for ensuring network redundancy?"): "Spanning Tree Protocol",

        # IT Support Specialist Questions and Answers
        ("Which helpdesk ticketing system do you use?"): "Zendesk",
        ("Which tool do you prefer for remote support?"): "TeamViewer",
        ("What is your preferred method for managing user access?"): "Active Directory",
        ("What documentation tool do you use for system issues?"): "Confluence",
        ("Which troubleshooting methodology do you follow?"): "Root cause analysis",
        ("What is the key to ensuring customer satisfaction?"): "Communication",

        # Web Developer Questions and Answers
        ("Which HTML version do you primarily work with?"): "HTML5",
        ("Which framework do you use for front-end development?"): "React",
        ("What layout technique do you use for responsive design?"): "Flexbox",
        ("What JavaScript method is used for asynchronous operations?"): "Promise",
        ("What tool do you use for web performance optimization?"): "Lighthouse",
        ("Which CSS preprocessor do you prefer?"): "SASS",

        # Product Manager Questions and Answers
        ("Which tool do you use for gathering requirements?"): "Jira",
        ("What method do you use to prioritize features?"): "MoSCoW",
        ("Which lifecycle stage is most critical for product success?"): "Development",
        ("Which team do you collaborate with most frequently?"): "Engineering",
        ("What project management methodology do you prefer?"): "Agile",
        ("What metric do you use to measure product success?"): "ROI",

        # Machine Learning Engineer Questions and Answers
        ("Which machine learning algorithm do you prefer?"): "Random Forest",
        ("What imputation method do you use for missing data?"): "Mean imputation",
        ("Which feature selection technique do you use?"): "PCA",
        ("What type of model did you use for your latest project?"): "Neural Network",
        ("Which deep learning framework do you use?"): "TensorFlow",
        ("What metric do you use to evaluate model performance?"): "Accuracy",

        # IT Project Manager Questions and Answers
        ("Which project management tool do you prefer?"): "Asana",
        ("What is your strategy for managing project risks?"): "Risk assessment",
        ("Which agile methodology do you follow?"): "Scrum",
        ("What is the most important factor in keeping your team motivated?"): "Communication",
        ("Which performance metric do you track in projects?"): "Velocity",
        ("Which software is most important for project documentation?"): "Confluence",

        # Business Analyst Questions and Answers
        ("Which method do you use to document business requirements?"): "Use cases",
        ("What tool do you prefer for gap analysis?"): "Excel",
        ("What is your preferred technique for resolving conflicting requirements?"): "Prioritization",
        ("Which process improvement methodology do you use?"): "Lean",
        ("Which tool do you use for creating wireframes?"): "Balsamiq",
        ("Which business metric is most critical to aligning solutions?"): "KPI",

        # Technical Support Engineer Questions and Answers
        ("Which troubleshooting method do you follow?"): "Divide and conquer",
        ("What tool do you use for handling escalated tickets?"): "Zendesk",
        ("What software do you use for remote troubleshooting?"): "AnyDesk",
        ("What term describes explaining technical concepts to non-technical users?"): "Jargon-free",
        ("What method do you use to resolve recurring issues?"): "Root cause analysis",
        ("Which resource keeps you updated on new troubleshooting techniques?"): "Tech blogs",

        # Quality Assurance Engineer Questions and Answers
        ("Which technique do you use to design test cases?"): "Boundary value analysis",
        ("What type of testing ensures no new issues are introduced?"): "Regression",
        ("Which prioritization method do you use for bugs?"): "Severity",
        ("Which automation tool do you use for testing?"): "Selenium",
        ("What method do you follow when critical bugs are found late?"): "Workaround",
        ("What is the term for collaboration between developers and testers?"): "DevTest",

        # Data Engineer Questions and Answers
        ("What is your preferred tool for designing data pipelines?"): "Apache Airflow",
        ("Which distributed processing system do you use?"): "Apache Spark",
        ("What technique do you use to ensure data integrity in ETL?"): "Checksum",
        ("Which platform do you use for real-time data processing?"): "Apache Kafka",
        ("Which cloud platform do you use for data engineering tasks?"): "AWS",
        ("What is your preferred method for securing sensitive data?"): "Encryption",

        # Artificial Intelligence Engineer Questions and Answers
        ("Which algorithm do you prefer for model training?"): "Gradient Boosting",
        ("What technique do you use to optimize neural networks?"): "Backpropagation",
        ("What project demonstrates your application of AI to real-world problems?"): "Recommendation system",
        ("Which AI framework do you use for development?"): "PyTorch",
        ("What is your approach to handling bias in AI models?"): "Fairness",
        ("What tool do you use to deploy AI models into production?"): "TensorFlow Serving",

        # UX/UI Designer Questions and Answers
        ("What method do you use for user research?"): "User Interviews",
        ("Which tool do you use for wireframing?"): "Figma",
        ("What usability improvement did you make in your last project?"): "Simplified navigation",
        ("What strategy do you use to balance user needs with business goals?"): "User-centered design",
        ("Which tool do you use for collaborating with developers?"): "Zeplin",
        ("What technique do you use to gather user feedback?"): "Surveys",

        # IT Consultant Questions and Answers
        ("Which methodology do you use for assessing IT infrastructure?"): "TOGAF",
        ("What project demonstrated your successful IT changes?"): "Digital Transformation of Legacy Systems",
        ("What methodology do you use for aligning IT projects with business goals?"): "Agile",
        ("Which resource keeps you updated with emerging technologies?"): "Tech Blogs, Conferences",
        ("What method do you use to manage resistance to IT changes?"): "Change Management",
        ("Which tool do you use for IT audits?"): "Qualys",

        # Solutions Architect Questions and Answers
        ("Which architecture framework do you follow?"): "TOGAF",
        ("What is your preferred method for designing scalable systems?"): "Microservices Architecture",
        ("Which cloud platform is your primary experience?"): "AWS",
        ("What factor influences your decisions on cost vs. technical trade-offs?"): "Business requirements",
        ("What solution did you design to solve a critical problem?"): "Cloud migration for disaster recovery",
        ("Which tool do you use to validate solution requirements?"): "Jira",

        # IT Operations Manager Questions and Answers
        ("What tool do you use to automate IT operations?"): "Ansible",
        ("Which metric measures system uptime?"): "Availability",
        ("What is the primary protocol for secure communication over a network?"): "HTTPS",
        ("Which standard governs IT service management processes?"): "ITIL",
        ("What is the most critical component in high-availability systems?"): "Redundancy",
        ("What method do you use for disaster recovery planning?"): "RTO (Recovery Time Objective)",

        # Chief Technology Officer (CTO) Questions and Answers
        ("Which framework is commonly used for technology strategy alignment?"): "TOGAF",
        ("What is the key measure of a technology’s scalability?"): "Throughput",
        ("Which cloud platform is known for its serverless computing?"): "AWS Lambda",
        ("What is the critical phase in digital transformation?"): "Integration",
        ("Which technology is used for enterprise architecture modeling?"): "ArchiMate",
        ("What is the most common database for unstructured data?"): "MongoDB",

        # Security Engineer Questions and Answers
        ("What is the most effective encryption algorithm for data security?"): "AES",
        ("What protocol is widely used for secure email communication?"): "PGP",
        ("What term refers to a network attack that floods a target with traffic?"): "DDoS",
        ("Which security framework is used to manage risk in an organization?"): "NIST",
        ("What technology is used for endpoint protection against malware?"): "EDR",
        ("What is the key standard for securing wireless networks?"): "WPA3",

        # IT Auditor Questions and Answers
        ("What is the most common framework used for IT governance?"): "COBIT",
        ("What standard is critical for evaluating information security?"): "ISO 27001",
        ("What is the first step in conducting a vulnerability assessment?"): "Scanning",
        ("Which tool is used to monitor system security events?"): "SIEM",
        ("What is the main objective of an IT audit?"): "Compliance",
        ("What is the key document produced during an IT audit?"): "Report",

        # Software Architect Questions and Answers
        ("What architectural style is used for microservices?"): "SOA (Service-Oriented Architecture)",
        ("What is the most widely used pattern for distributed systems?"): "Event-driven",
        ("Which protocol is commonly used for real-time web applications?"): "WebSocket",
        ("What design pattern is best for decoupling object creation?"): "Factory",
        ("What database model is ideal for hierarchical data?"): "NoSQL",
        ("What technique is used to ensure fault tolerance in a distributed system?"): "Replication",

        # Scrum Master Questions and Answers
        ("What agile methodology is used for managing iterative project work?"): "Scrum",
        ("What is the typical timebox for a sprint?"): "Two weeks",
        ("Which metric is used to measure team performance in Scrum?"): "Velocity",
        ("What tool is commonly used for agile project tracking?"): "Jira",
        ("What is the primary role of the Scrum Master?"): "Facilitation",
        ("What is the Scrum event where the team plans work for the next sprint?"): "Sprint Planning",

        # Technical Writer Questions and Answers
        ("What markup language is used to write documentation for technical content?"): "Markdown",
        ("What process do you use to organize large sets of documentation?"): "Structuring",
        ("Which tool is preferred for creating and editing technical content?"): "Confluence",
        ("What document type is commonly used for API documentation?"): "Swagger",
        ("Which standard defines the style and formatting of technical documents?"): "Chicago",
        ("What writing technique is essential for simplifying complex information?"): "Clarity",

        # Network Security Analyst Questions and Answers
        ("Which protocol is used for securing email communication?"): "SMTP",
        ("What tool is commonly used for network vulnerability scanning?"): "Nessus",
        ("What technology provides encrypted tunnels for secure communication?"): "VPN",
        ("Which algorithm is commonly used for public-key encryption?"): "RSA",
        ("What is the standard for network security monitoring?"): "SNMP",
        ("Which attack attempts to overwhelm a network with traffic?"): "Flooding",

        # Game Developer Questions and Answers
        ("Which game engine is widely used for 3D game development?"): "Unity",
        ("What technology is used for physics simulation in games?"): "Havok",
        ("Which language is commonly used for scripting in Unreal Engine?"): "Blueprints",
        ("What technique is used to reduce the file size of game textures?"): "Compression",
        ("What method is used to synchronize multiplayer game states?"): "RPC (Remote Procedure Call)",
        ("What is the most common data structure for representing a game world?"): "Graph",

        # Embedded Systems Engineer Questions and Answers
        ("Which protocol is commonly used for communication in embedded systems?"): "SPI",
        ("What is the primary language used for embedded software development?"): "C",
        ("What tool is commonly used for debugging embedded systems?"): "JTAG",
        ("What is the most common architecture for embedded systems?"): "ARM",
        ("Which operating system is used for real-time embedded systems?"): "RTOS",
        ("What is the primary concern for embedded systems in battery-powered devices?"): "Power",

        # ERP Consultant Questions and Answers
        ("What ERP platform is widely used for large enterprises?"): "SAP",
        ("Which methodology is often used for ERP system implementation?"): "ASAP",
        ("What tool do you use for ERP data migration?"): "LSMW",
        ("What is the most important factor in ERP customization?"): "Flexibility",
        ("Which programming language is used for customizing SAP ERP?"): "ABAP",
        ("What is the standard for ERP system integration?"): "Middleware",

        # Salesforce Developer Questions and Answers
        ("What is the primary language used for custom development in Salesforce?"): "Apex",
        ("Which framework is commonly used for building UI in Salesforce?"): "Visualforce",
        ("What is the most commonly used tool for Salesforce data integration?"): "MuleSoft",
        ("Which method is used to query data in Salesforce?"): "SOQL",
        ("What type of cloud is Salesforce primarily known for?"): "CRM",
        ("What Salesforce product is used for customer service management?"): "ServiceCloud",

        # Big Data Engineer Questions and Answers
        ("What framework is most commonly used for distributed data processing?"): "Hadoop",
        ("Which programming language is typically used for Spark applications?"): "Scala",
        ("What is the default storage format in Hadoop?"): "HDFS",
        ("Which tool is most commonly used for real-time big data processing?"): "Kafka",
        ("What is the main advantage of using NoSQL databases for big data?"): "Scalability",
        ("Which tool is used for batch processing in big data systems?"): "Hive",

        # BI Developer Questions and Answers
        ("Which BI tool is commonly used for data visualization?"): "Tableau",
        ("What query language is used in Power BI for data modeling?"): "DAX",
        ("What is the most common format for storing BI data?"): "OLAP",
        ("Which database system is typically used for BI reporting?"): "SQL Server",
        ("What is the key principle of the ETL process?"): "Extraction",
        ("Which BI tool is known for its integration with Excel?"): "Power BI",

        # Information Security Analyst Questions and Answers
        ("What tool is commonly used for vulnerability scanning?"): "Nessus",
        ("What encryption algorithm is widely used for securing web traffic?"): "AES",
        ("Which standard is used for information security management?"): "ISO 27001",
        ("What is the most common method for network intrusion detection?"): "IDS",
        ("Which attack method involves overwhelming a system with requests?"): "DDoS",
        ("What is the best practice for securing passwords?"): "Hashing",

        # Robotics Engineer Questions and Answers
        ("What programming language is commonly used for robotic systems?"): "C++",
        ("Which software framework is used for controlling robots?"): "ROS",
        ("What sensor is most commonly used for measuring distance in robots?"): "LIDAR",
        ("What is the key factor for real-time performance in robotics?"): "Latency",
        ("Which algorithm is used for robot path planning?"): "A*",
        ("What is the primary challenge in autonomous robotics?"): "Localization",

        # Cloud Solutions Architect Questions and Answers
        ("What is the most widely used cloud provider?"): "AWS",
        ("Which cloud service model provides the highest level of control?"): "IaaS",
        ("What is the most popular cloud storage solution?"): "S3",
        ("Which protocol is used for secure communication in cloud environments?"): "HTTPS",
        ("What technology enables automatic scaling of cloud applications?"): "Autoscaling",
        ("What is the main tool for managing cloud infrastructure as code?"): "Terraform",

        # Computer Vision Engineer Questions and Answers
        ("Which framework is commonly used for deep learning in computer vision?"): "TensorFlow",
        ("What technique is used for object detection in images?"): "CNN",
        ("What is the most common metric for evaluating object detection performance?"): "mAP",
        ("What algorithm is used for facial recognition?"): "LBP",
        ("Which tool is widely used for image processing in computer vision?"): "OpenCV",
        ("What technique is used to handle lighting variations in images?"): "Histogram",

        # Site Reliability Engineer Questions and Answers
        ("What is the main tool used for infrastructure automation?"): "Ansible",
        ("Which service is used for monitoring distributed systems?"): "Prometheus",
        ("What is the term for managing server configurations in code?"): "IaC",
        ("Which metric is critical for evaluating system reliability?"): "SLA",
        ("What is the most common tool for continuous integration in DevOps?"): "Jenkins",
        ("Which tool is used for container orchestration?"): "Kubernetes",

        # Penetration Tester Questions and Answers
        ("What is the primary tool for scanning network vulnerabilities?"): "Nmap",
        ("Which attack method is used for gaining unauthorized access to a system?"): "Exploitation",
        ("What is the best tool for password cracking?"): "John",
        ("Which protocol is most often targeted in network penetration testing?"): "SMB",
        ("What is the most commonly used technique for web application testing?"): "SQL Injection",
        ("What type of penetration test is conducted with permission from the system owner?"): "Ethical",

        # Data Analyst Questions and Answers
        ("Which tool is most commonly used for data analysis in businesses?"): "Excel",
        ("What is the most common language for data manipulation?"): "Python",
        ("What is the most important aspect of data cleaning?"): "Consistency",
        ("Which statistical test is commonly used for hypothesis testing?"): "T-test",
        ("What is the most common format for structured data?"): "CSV",
        ("What is the most widely used database for data analysis?"): "SQL",

        # Blockchain Developer Questions and Answers
        ("Which blockchain platform is known for its smart contract functionality?"): "Ethereum",
        ("What language is most commonly used for writing Ethereum smart contracts?"): "Solidity",
        ("Which consensus algorithm does Bitcoin use?"): "Proof-of-Work",
        ("What tool is commonly used for testing Ethereum contracts?"): "Truffle",
        ("What is the most popular blockchain for enterprise use?"): "Hyperledger",
        ("Which type of blockchain ensures complete transparency and immutability?"): "Public",

        # IT Compliance Specialist Questions and Answers
        ("Which regulation is focused on data protection and privacy in the EU?"): "GDPR",
        ("What framework is most commonly used for IT compliance in security?"): "ISO 27001",
        ("Which standard is widely adopted for IT service management?"): "ITIL",
        ("What is the most common audit framework used in IT?"): "SOC 2",
        ("Which tool is commonly used for IT compliance monitoring?"): "GRC",
        ("What is the most critical component in maintaining IT compliance?"): "Documentation",

        # Software Development Manager Questions and Answers
        ("Which methodology emphasizes iterative development and collaboration?"): "Agile",
        ("What tool is commonly used for version control in software development?"): "Git",
        ("Which programming language is known for its use in enterprise software?"): "Java",
        ("What is the main concept behind DevOps?"): "Automation",
        ("What is the most common model for project management in software development?"): "Scrum",
        ("What is the primary responsibility of a software development manager?"): "Leadership",

        # Virtual Reality Developer Questions and Answers
        ("Which game engine is commonly used for VR development?"): "Unity",
        ("What is the most common VR hardware used for immersive experiences?"): "Oculus",
        ("Which technology is commonly integrated with VR for real-time interactivity?"): "AI",
        ("What is the most important consideration in designing VR user interfaces?"): "Comfort",
        ("What is the primary challenge in VR content creation?"): "Motion sickness",
        ("Which programming language is most commonly used for VR development?"): "C#",

        # Site Reliability Engineer Questions and Answers
        ("Which tool is commonly used for monitoring system performance?"): "Prometheus",
        ("What methodology focuses on the reliability and performance of services?"): "SRE",
        ("What is the most common metric for measuring service availability?"): "Uptime",
        ("What is the most commonly used technique for automating infrastructure?"): "Terraform",
        ("What tool is used for container orchestration in cloud environments?"): "Kubernetes",
        ("What is the primary focus of a site reliability engineer?"): "Reliability",

        # Infrastructure Engineer Questions and Answers
        ("What cloud platform is most commonly used for building infrastructure?"): "AWS",
        ("What is the most common virtualization technology for managing servers?"): "VMware",
        ("Which network protocol is most commonly used for securing communications?"): "HTTPS",
        ("What is the most important factor when designing scalable infrastructure?"): "Load balancing",
        ("Which system is used for infrastructure automation?"): "Ansible",
        ("What type of network is used to connect cloud-based resources?"): "VPC",

        # IT Operations Analyst Questions and Answers
        ("What is the most common framework for IT service management?"): "ITIL",
        ("Which tool is widely used for incident management?"): "ServiceNow",
        ("What is the key goal of IT operations management?"): "Efficiency",
        ("What process is used to restore normal service after an incident?"): "Incident resolution",
        ("What is the primary goal of a monitoring tool in IT operations?"): "Uptime",
        ("What is the most important aspect of a post-incident review?"): "Analysis",

        # Digital Marketing Specialist Questions and Answers
        ("What is the most common metric for measuring website traffic?"): "Visits",
        ("Which tool is commonly used for SEO analysis?"): "Google Analytics",
        ("What is the most common social media platform for B2B marketing?"): "LinkedIn",
        ("Which advertising model is based on pay-per-click?"): "PPC",
        ("What is the primary goal of content marketing?"): "Engagement",
        ("What is the most common format for digital marketing ads?"): "Display",

        # Network Architect Questions and Answers
        ("Which protocol is most commonly used for routing in networks?"): "BGP",
        ("What technology is used to virtualize network resources?"): "SDN",
        ("Which type of network is typically used to interconnect data centers?"): "WAN",
        ("What is the most important factor when designing secure network architectures?"): "Firewalls",
        ("Which tool is commonly used for network performance monitoring?"): "Wireshark",
        ("What is the primary goal of a network architect?"): "Scalability",

        # Help Desk Technician Questions and Answers
        ("Which tool is widely used for ticketing and issue tracking?"): "Jira",
        ("What is the first step in troubleshooting an IT issue?"): "Diagnosis",
        ("Which operating system is most commonly used in enterprise IT environments?"): "Windows",
        ("What method is commonly used to solve recurring IT issues?"): "Documentation",
        ("What is the primary role of a help desk technician?"): "Support",
        ("Which service is typically used for remote IT support?"): "TeamViewer",
   
        # Configuration Manager Questions and Answers
        ("What is the most commonly used tool for configuration management?"): "Ansible",
        ("Which version control system is most widely used in IT?"): "Git",
        ("What is the primary goal of configuration management?"): "Consistency",
        ("Which framework is most commonly used for change management?"): "ITIL",
        ("What is the most critical factor in maintaining configuration standards?"): "Auditing",
        ("What is the primary challenge in configuration management?"): "Complexity",

        # Systems Analyst Questions and Answers
        ("Which tool is commonly used for process modeling?"): "BPMN",
        ("What method is most commonly used for requirements gathering?"): "Interviews",
        ("What is the primary focus of a systems analyst?"): "Optimization",
        ("Which software is commonly used for systems integration testing?"): "Selenium",
        ("What is the most important factor when analyzing system performance?"): "Efficiency",
        ("What is the most important consideration when implementing a new system?"): "Scalability",

        # Database Developer Questions and Answers
        ("Which SQL command is used to retrieve data from a database?"): "SELECT",
        ("What is the most important factor in database optimization?"): "Indexing",
        ("Which programming language is most commonly used for database procedures?"): "PL/SQL",
        ("What is the most common type of database used in enterprise applications?"): "Relational",
        ("Which tool is commonly used for database management?"): "MySQL",
        ("What is the key principle behind database normalization?"): "Minimization",

        # IT Business Partner Questions and Answers
        ("Which framework is commonly used to align IT and business strategies?"): "COBIT",
        ("What is the most common metric for measuring IT performance?"): "ROI",
        ("Which process is most important for managing IT resources?"): "Budgeting",
        ("What is the primary role of an IT business partner?"): "Alignment",
        ("What is the most important factor when identifying digital transformation opportunities?"): "Innovation",
        ("Which tool is commonly used for IT project tracking?"): "Jira",

        # Cloud Consultant Questions and Answers
        ("Which cloud platform is most widely used for enterprise applications?"): "AWS",
        ("What is the primary advantage of cloud computing?"): "Scalability",
        ("Which service is commonly used for serverless computing?"): "Lambda",
        ("What is the most important factor when designing a cloud architecture?"): "Flexibility",
        ("Which tool is commonly used for managing cloud costs?"): "CloudWatch",
        ("What is the most important aspect of cloud security?"): "Encryption",

        # Virtualization Engineer Questions and Answers
        ("What is the most widely used virtualization platform?"): "VMware",
        ("Which technology is often used to containerize applications?"): "Docker",
        ("What is the primary benefit of virtualization?"): "Efficiency",
        ("Which type of virtualization is commonly used for server management?"): "Hypervisor",
        ("What is the most important factor when optimizing a virtualized environment?"): "Resource allocation",
        ("Which tool is most commonly used for managing virtual machines?"): "vSphere",

        # E-commerce Specialist Questions and Answers
        ("Which e-commerce platform is most widely used for online stores?"): "Shopify",
        ("What is the most important factor in increasing online sales?"): "Conversion",
        ("Which metric is most commonly used to measure e-commerce success?"): "Revenue",
        ("What is the most common payment gateway used in e-commerce?"): "PayPal",
        ("What is the key strategy for optimizing e-commerce websites?"): "UX/UI",
        ("Which tool is commonly used to track e-commerce metrics?"): "Google Analytics",

        # IT Trainer Questions and Answers
        ("Which platform is commonly used for online IT training?"): "Udemy",
        ("What is the most important factor when designing training programs?"): "Engagement",
        ("What tool is widely used for virtual classrooms?"): "Zoom",
        ("What is the most common method for assessing training effectiveness?"): "Feedback",
        ("Which technique is most effective for teaching complex IT concepts?"): "Hands-on",
        ("What is the most important consideration when customizing training materials?"): "Audience",

        # Technical Project Manager Questions and Answers
        ("Which methodology is most commonly used for managing technical projects?"): "Agile",
        ("What is the primary responsibility of a technical project manager?"): "Delivery",
        ("Which tool is most widely used for project tracking?"): "Jira",
        ("What is the most common method for managing project risks?"): "Mitigation",
        ("Which project management process is most crucial for scope management?"): "Change control",
        ("What is the most important aspect when managing a project budget?"): "Accuracy",

        # Mobile UX Designer Questions and Answers
        ("Which tool is most commonly used for mobile wireframing?"): "Figma",
        ("What is the most important aspect of mobile UX design?"): "Simplicity",
        ("Which platform is most commonly used for mobile app testing?"): "TestFlight",
        ("What is the primary challenge in designing mobile interfaces?"): "Screen size",
        ("Which design principle is crucial for mobile UX?"): "Responsiveness",
        ("What is the most important factor in mobile app usability?"): "Navigation",
     
        # Network Operations Center (NOC) Technician Questions and Answers
        ("What is the most commonly used tool for network monitoring?"): "Nagios",
        ("What is the first step in troubleshooting network issues?"): "Diagnosis",
        ("Which protocol is primarily used for remote management of network devices?"): "SNMP",
        ("What is the key principle behind network troubleshooting?"): "Isolation",
        ("What is the most important factor when prioritizing network incidents?"): "Severity",
        ("What is the most common network maintenance task?"): "Monitoring",

        # Release Manager Questions and Answers
        ("What tool is most commonly used for version control in software development?"): "Git",
        ("What is the primary goal of a release manager?"): "Coordination",
        ("What process is used to ensure a smooth deployment?"): "Automation",
        ("Which framework is widely used for continuous integration?"): "Jenkins",
        ("What is the most critical factor in a successful release?"): "Communication",
        ("What tool is used for tracking post-release feedback?"): "Jira",

        # IT Change Manager Questions and Answers
        ("What framework is commonly used for IT change management?"): "ITIL",
        ("What tool is most commonly used for change request tracking?"): "ServiceNow",
        ("What is the primary objective of change management?"): "Control",
        ("Which process is used to assess the impact of changes?"): "Risk assessment",
        ("What is the most important factor in managing IT changes?"): "Communication",
        ("What is the most common type of IT change?"): "Updates",

        # Data Governance Analyst Questions and Answers
        ("What is the most important regulation for data protection?"): "GDPR",
        ("What is the most commonly used tool for data governance?"): "Collibra",
        ("What is the primary goal of data governance?"): "Integrity",
        ("Which data management practice is crucial for ensuring compliance?"): "Auditing",
        ("What is the key focus when establishing data governance policies?"): "Consistency",
        ("What is the first step in managing data lineage?"): "Mapping",

        # Performance Engineer Questions and Answers
        ("Which tool is most commonly used for load testing?"): "JMeter",
        ("What metric is most critical for evaluating system performance?"): "Latency",
        ("What is the most important factor when addressing performance bottlenecks?"): "Profiling",
        ("Which technology is commonly used for performance monitoring?"): "Prometheus",
        ("What is the key metric when evaluating application performance under heavy load?"): "Throughput",
        ("What is the primary focus of performance tuning?"): "Optimization",

        # BI Analyst Questions and Answers
        ("Which BI tool is most widely used for data visualization?"): "Tableau",
        ("What metric is most crucial for evaluating BI effectiveness?"): "Accuracy",
        ("Which process is essential when validating data accuracy?"): "Cleansing",
        ("What type of analysis is commonly used to interpret BI data?"): "Trend",
        ("What is the most important factor in aligning BI solutions with business goals?"): "Relevance",
        ("What is the most common method for gathering business requirements?"): "Interviews",

        # SAP Consultant Questions and Answers
        ("Which SAP module is most commonly implemented in finance?"): "FICO",
        ("What tool is widely used for SAP system troubleshooting?"): "Solution Manager",
        ("What is the key benefit of using SAP S/4HANA?"): "Speed",
        ("Which process is most critical for successful SAP implementation?"): "Configuration",
        ("What is the most important factor in ensuring SAP system integration?"): "Compatibility",
        ("Which type of SAP system is primarily used for data migration?"): "ETL",

        # Digital Transformation Consultant Questions and Answers
        ("Which technology is most commonly used in digital transformation?"): "Cloud",
        ("What is the most critical aspect when managing digital transformation?"): "Change",
        ("Which framework is widely used to assess digital maturity?"): "DMM",
        ("What is the most important factor in ensuring alignment with business goals?"): "Strategy",
        ("Which KPI is most commonly used to measure digital transformation success?"): "ROI",
        ("What method is commonly used to manage resistance to digital change?"): "Communication",

        # IT Asset Manager Questions and Answers
        ("What tool is most commonly used for IT asset tracking?"): "ServiceNow",
        ("What is the primary goal of IT asset management?"): "Optimization",
        ("Which process is most critical for managing software licenses?"): "Auditing",
        ("What is the most important factor in managing asset lifecycles?"): "Planning",
        ("Which action is essential when disposing of IT assets?"): "Sanitization",
        ("What is the primary focus when improving IT asset management efficiency?"): "Automation",

        # Game Designer Questions and Answers
        ("Which game engine is most commonly used for 3D game design?"): "Unreal",
        ("What is the most important aspect of gameplay design?"): "Engagement",
        ("Which tool is most commonly used for prototyping game concepts?"): "Unity",
        ("What is the most critical factor in balancing game design?"): "Playability",
        ("Which programming language is widely used in game development?"): "C#",
        ("What is the most important element in creating a game narrative?"): "Storytelling",

        # Social Media Analyst Questions and Answers
        ("What tool is most commonly used for social media performance analysis?"): "Hootsuite",
        ("What metric is most important for measuring social media ROI?"): "Engagement",
        ("Which platform is most critical for tracking brand sentiment?"): "Twitter",
        ("What method is widely used for audience segmentation?"): "Demographics",
        ("Which type of analysis is crucial for improving social media engagement?"): "A/B testing",
        ("What is the key factor in analyzing competitor performance on social media?"): "Benchmarking",
    }
