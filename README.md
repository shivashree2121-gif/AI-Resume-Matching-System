
# 🤖 AI Resume–Job Matching System

An NLP-based Resume Matching System that compares candidate resumes with a job description, calculates an overall matching score, identifies matched and missing skills, and ranks multiple candidates based on their suitability for the role.

## 📌 Project Overview

Recruiters often receive a large number of resumes for a single job opening. Manually reviewing every resume can be time-consuming.

This project automates the initial resume screening process using Natural Language Processing and Machine Learning techniques.

The system analyzes resumes against a job description and provides:

- Resume-to-job similarity score
- Skill match percentage
- Matched skills
- Missing skills
- Overall candidate score
- Candidate ranking
- Candidate recommendations

## 🚀 Features

- 📄 Upload PDF and DOCX resumes
- 🧹 Resume text preprocessing
- 🧠 TF-IDF text vectorization
- 🔍 Cosine similarity
- 🛠️ Automated skill extraction
- ✅ Matched skill identification
- ❌ Missing skill identification
- 🎯 Weighted overall match score
- 🏆 Multiple candidate ranking
- 📊 Candidate comparison
- 🌐 Streamlit web application

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| NLTK | Natural Language Processing |
| Scikit-learn | TF-IDF and Cosine Similarity |
| PyPDF | PDF text extraction |
| python-docx | DOCX text extraction |
| Streamlit | Web application |

## ⚙️ System Workflow

```text
Resume PDF / DOCX
        ↓
Text Extraction
        ↓
Text Cleaning
        ↓
TF-IDF Vectorization
        ↓
Cosine Similarity
        ↓
Skill Extraction
        ↓
Skill Matching
        ↓
Weighted Match Score
        ↓
Candidate Ranking
        ↓
Streamlit Dashboard
