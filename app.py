import re
import string

import streamlit as st
import pandas as pd
import nltk

from pypdf import PdfReader
from docx import Document

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# NLTK SETUP
# ============================================================

nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# SKILL DATABASE
# ============================================================

skills = [
    # Programming
    "python",
    "java",
    "r",
    "c++",

    # Databases
    "sql",
    "mysql",
    "postgresql",
    "mongodb",

    # Data Analysis
    "excel",
    "power bi",
    "tableau",
    "pandas",
    "numpy",

    # Machine Learning
    "machine learning",
    "deep learning",
    "scikit-learn",
    "tensorflow",
    "pytorch",

    # Statistics
    "statistics",
    "hypothesis testing",
    "regression",
    "probability",

    # Data / AI
    "data analysis",
    "data visualization",
    "nlp",
    "artificial intelligence",

    # Cloud
    "aws",
    "azure",
    "google cloud",

    # Tools
    "git",
    "github",
    "docker"
]


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def extract_docx_text(uploaded_file):
    document = Document(uploaded_file)

    text = ""

    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"

    return text


def extract_text(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_pdf_text(uploaded_file)

    elif file_name.endswith(".docx"):
        return extract_docx_text(uploaded_file)

    else:
        return ""


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        " ",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Split into words
    words = text.split()

    # Remove stopwords
    words = [
        word
        for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(words)


# ============================================================
# SKILL EXTRACTION
# ============================================================

def extract_skills(text):

    text = text.lower()

    found_skills = []

    for skill in skills:

        if skill in text:
            found_skills.append(skill)

    return sorted(set(found_skills))


# ============================================================
# RESUME MATCHING
# ============================================================

def calculate_candidate_score(resume, job):

    # Clean text
    resume_clean = clean_text(resume)
    job_clean = clean_text(job)

    # TF-IDF
    vectorizer = TfidfVectorizer()

    matrix = vectorizer.fit_transform(
        [
            resume_clean,
            job_clean
        ]
    )

    # Cosine similarity
    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )[0][0] * 100

    # Extract skills
    resume_skills = extract_skills(resume)
    job_skills = extract_skills(job)

    # Matched skills
    matched = sorted(
        set(resume_skills) &
        set(job_skills)
    )

    # Missing skills
    missing = sorted(
        set(job_skills) -
        set(resume_skills)
    )

    # Skill score
    if len(job_skills) > 0:

        skill_score = (
            len(matched) /
            len(job_skills)
        ) * 100

    else:

        skill_score = 0

    # Final weighted score
    final_score = (
        similarity * 0.60 +
        skill_score * 0.40
    )

    return {
        "Text Similarity": round(similarity, 2),
        "Skill Match": round(skill_score, 2),
        "Final Score": round(final_score, 2),
        "Matched Skills": matched,
        "Missing Skills": missing
    }


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.title("📄 AI Resume–Job Matching System")

st.write(
    "Upload multiple resumes and compare them "
    "against a job description using NLP and "
    "Machine Learning."
)

st.divider()


# ============================================================
# UPLOAD RESUMES
# ============================================================

st.subheader("📂 Upload Resumes")

uploaded_resumes = st.file_uploader(
    "Upload candidate resumes",
    type=["pdf", "docx"],
    accept_multiple_files=True
)


# ============================================================
# JOB DESCRIPTION
# ============================================================

st.subheader("💼 Job Description")

job_description = st.text_area(
    "Paste the job description here:",
    height=250
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🚀 Analyze Candidates",
    type="primary"
):

    if not uploaded_resumes:

        st.error(
            "Please upload at least one resume."
        )

    elif not job_description.strip():

        st.error(
            "Please enter a job description."
        )

    else:

        results = []

        detailed_results = {}

        # ====================================================
        # PROCESS EACH RESUME
        # ====================================================

        for resume_file in uploaded_resumes:

            try:

                resume_text = extract_text(
                    resume_file
                )

                if not resume_text.strip():

                    st.warning(
                        f"Could not extract text from "
                        f"{resume_file.name}"
                    )

                    continue

                result = calculate_candidate_score(
                    resume_text,
                    job_description
                )

                results.append({

                    "Resume":
                        resume_file.name,

                    "Text Similarity":
                        result["Text Similarity"],

                    "Skill Match":
                        result["Skill Match"],

                    "Final Score":
                        result["Final Score"]

                })

                detailed_results[
                    resume_file.name
                ] = result

            except Exception as e:

                st.error(
                    f"Error processing "
                    f"{resume_file.name}: {e}"
                )

        # ====================================================
        # RANKING
        # ====================================================

        if results:

            ranking_df = pd.DataFrame(
                results
            )

            ranking_df = ranking_df.sort_values(
                by="Final Score",
                ascending=False
            ).reset_index(drop=True)

            ranking_df.index += 1

            st.divider()

            # =================================================
            # TOP CANDIDATE
            # =================================================

            top_candidate = ranking_df.iloc[0]

            st.subheader("🏆 Best Candidate")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Candidate",
                top_candidate["Resume"]
            )

            col2.metric(
                "Overall Match",
                f'{top_candidate["Final Score"]:.2f}%'
            )

            col3.metric(
                "Skill Match",
                f'{top_candidate["Skill Match"]:.2f}%'
            )

            # =================================================
            # RANKING TABLE
            # =================================================

            st.subheader("📊 Candidate Ranking")

            st.dataframe(
                ranking_df,
                use_container_width=True
            )

            # =================================================
            # CHART
            # =================================================

            st.subheader("📈 Match Score Comparison")

            chart_data = ranking_df.set_index(
                "Resume"
            )[["Final Score"]]

            st.bar_chart(
                chart_data
            )

            # =================================================
            # CANDIDATE DETAILS
            # =================================================

            st.subheader(
                "🔍 Candidate Skill Analysis"
            )

            selected_candidate = st.selectbox(
                "Select a candidate:",
                ranking_df["Resume"].tolist()
            )

            selected_result = detailed_results[
                selected_candidate
            ]

            col1, col2 = st.columns(2)

            # =================================================
            # MATCHED SKILLS
            # =================================================

            with col1:

                st.markdown(
                    "### ✅ Matched Skills"
                )

                if selected_result[
                    "Matched Skills"
                ]:

                    for skill in selected_result[
                        "Matched Skills"
                    ]:

                        st.write(
                            f"✓ {skill}"
                        )

                else:

                    st.write(
                        "No matching skills found."
                    )

            # =================================================
            # MISSING SKILLS
            # =================================================

            with col2:

                st.markdown(
                    "### ❌ Missing Skills"
                )

                if selected_result[
                    "Missing Skills"
                ]:

                    for skill in selected_result[
                        "Missing Skills"
                    ]:

                        st.write(
                            f"✗ {skill}"
                        )

                else:

                    st.write(
                        "No major missing skills."
                    )

            # =================================================
            # RECOMMENDATION
            # =================================================

            score = selected_result[
                "Final Score"
            ]

            st.subheader(
                "💡 Recommendation"
            )

            if score >= 80:

                st.success(
                    "Excellent Match — "
                    "Strongly Recommended"
                )

            elif score >= 65:

                st.info(
                    "Good Match — Recommended"
                )

            elif score >= 50:

                st.warning(
                    "Moderate Match — "
                    "Consider Improving Skills"
                )

            else:

                st.error(
                    "Low Match — "
                    "Significant Skill Gaps"
                )

        else:

            st.warning(
                "No valid resumes could be analyzed."
            )
