import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
@st.cache_resource

def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def extract_relevant_resume_sections(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text() or ""
        
    text_upper = full_text.upper()
    start_idx = -1
    for kw in ["TECHNICAL SKILLS", "PROJECTS", "INTERNSHIP EXPERIENCE"]:
        idx = text_upper.find(kw)
        if idx != -1 and (start_idx == -1 or idx < start_idx):
            start_idx = idx
            
    
    end_idx = text_upper.find("EDUCATION")
    
    if start_idx != -1:
        if end_idx != -1 and end_idx > start_idx:
            relevant_text = full_text[start_idx:end_idx]
        else:
            relevant_text = full_text[start_idx:]
    else:
        relevant_text = full_text # Fallback if headers aren't found
        
    return relevant_text

st.set_page_config(page_title="AI Resume Matcher", page_icon="📄", layout="centered")

st.title("📄 Smart Resume & Job Description Semantic Matcher")
st.write("Upload your resume and paste the job description to see how well you match!")

st.header("1. Upload Your Resume")
uploaded_file = st.file_uploader("Upload your resume in PDF format", type=["pdf"])

st.header("2. Paste the Job Description")
job_description = st.text_area("Paste the job description text here:", height=200)

if st.button("Calculate Match"):
    if uploaded_file is not None and job_description.strip() != "":
        with st.spinner("Extracting technical sections and calculating AI semantic match... Please wait."):
            
            resume_text = extract_relevant_resume_sections(uploaded_file)
            
            if not resume_text.strip():
                st.error("Could not extract any text from this PDF. Make sure it's not a scanned image!")
            else:
                resume_embedding = model.encode([resume_text])
                job_embedding = model.encode([job_description])
                
                similarity_score = cosine_similarity(resume_embedding, job_embedding)[0][0]
                match_percentage = round(float(similarity_score) * 100, 1)
            
                st.success("Analysis Complete!")
                st.metric(label="Focused Technical Match Score", value=f"{match_percentage}%")
                
                if match_percentage >= 75:
                    st.balloons()
                    st.success("🌟 **Great match!** Your technical skills and projects strongly align with this job description.")
                elif match_percentage >= 50:
                    st.info("👍 **Moderate match.** You have relevant technical experience, but consider highlighting more target keywords.")
                else:
                    st.warning("⚠️ **Low match.** This role might require different tech stacks than what is in your projects/skills.")
                
                st.subheader("🔍 Quick Keyword Insight")
                resume_lower = resume_text.lower()
                job_lower = job_description.lower()
                
                common_keywords = ["python", "machine learning", "sql", "aws", "docker", "fastapi", "react", "git", "java", "pandas", "iot", "html"]
                missing_skills = [kw.capitalize() for kw in common_keywords if kw in job_lower and kw not in resume_lower]
                
                if missing_skills:
                    st.write("Suggested skills found in the job description that could be added to your resume:")
                    st.write(", ".join(missing_skills))
                else:
                    st.write("No major standard tech keyword gaps detected!")

                st.subheader("⚖️ Job Description Bias Analyzer")
                biased_words = ["ninja", "rockstar", "guru", "aggressive", "dominate", "fast-paced", "digital native"]
                found_bias = [word for word in biased_words if word in job_lower]
                
                if found_bias:
                    st.warning(f"Found potentially exclusive or overly aggressive buzzwords: **{', '.join(found_bias)}**. Consider replacing them for a more inclusive job post.")
                else:
                    st.success("The job description uses clean, inclusive language with no common exclusionary buzzwords detected!")
                    
    else:
        st.warning("Please upload a resume PDF and paste a job description first!")