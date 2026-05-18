import streamlit as st
import nltk
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# ---------------- FUNCTIONS ---------------- #

def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def calculate_similarity(resume_text, job_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, job_text])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(similarity * 100, 2)

def extract_skills(text):
    skills = [
        "python", "machine learning", "nlp", "sql",
        "data analysis", "deep learning", "tensorflow",
        "pandas", "numpy", "scikit-learn"
    ]
    found = [skill for skill in skills if skill.lower() in text.lower()]
    return found

# ---------------- UI CONFIG ---------------- #

st.set_page_config(page_title="AI Resume Screener", page_icon="🤖", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #4CAF50;
    }
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #aaaaaa;
        margin-bottom: 30px;
    }
    .card {
        padding: 25px;
        border-radius: 15px;
        background-color: #1c1f26;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #

st.markdown('<div class="title">🤖 AI Resume Screening System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Match your resume with job descriptions using NLP</div>', unsafe_allow_html=True)

# ---------------- MAIN CARD ---------------- #

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    resume_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
    job_file = st.file_uploader("📑 Upload Job Description (PDF or TXT)", type=["pdf", "txt"])

    if resume_file and job_file:

        with st.spinner("🔍 Analyzing documents..."):
            time.sleep(1.5)

            resume_text = extract_text_from_pdf(resume_file)

            if job_file.type == "application/pdf":
                job_text = extract_text_from_pdf(job_file)
            else:
                job_text = job_file.read().decode("utf-8")

            score = calculate_similarity(resume_text, job_text)
            skills = extract_skills(resume_text)

        # ---------------- RESULTS ---------------- #

        st.markdown("### 📊 Match Score")

        # Progress bar
        st.progress(int(score))

        # Color-coded score
        if score >= 70:
            st.success(f"✅ Strong Match: {score}%")
        elif score >= 40:
            st.warning(f"⚠️ متوسط Match: {score}%")
        else:
            st.error(f"❌ Low Match: {score}%")

        # Skills section
        st.markdown("### 🧠 Skills Detected")

        if skills:
            st.write(", ".join(skills))
        else:
            st.write("No relevant skills detected")

        # Suggestions
        st.markdown("### 💡 Suggestions")

        if score < 70:
            st.info("Add more relevant skills from the job description to improve your score.")
        else:
            st.success("Your resume is well aligned with this job!")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #

st.markdown("---")
