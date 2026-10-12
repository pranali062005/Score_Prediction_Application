import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Page configuration
st.set_page_config(page_title="Student Performance Predictor", page_icon="📊", layout="centered")

# Load the KNeighborsRegressor model safely
@st.cache_resource
def load_model():
    model_path = 'knn.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file '{model_path}' was not found. Please ensure 'kn.pkl' is uploaded to your repository.")
    with open(model_path, 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Student Performance Predictor")
st.write("Enter the student details below to generate predictions using the trained K-Nearest Neighbors model[cite: 1].")

with st.form("prediction_form"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=5, max_value=100, value=20)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        course = st.text_input("Course", "General Studies")
        study_hours = st.number_input("Study Hours", min_value=0.0, max_value=168.0, value=10.0)
        class_attendance = st.number_input("Class Attendance (%)", min_value=0.0, max_value=100.0, value=80.0)
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
        
    with col2:
        sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
        sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Average", "Good", "Excellent"])
        study_method = st.selectbox("Study Method", ["Self-Study", "Online", "Group", "Hybrid"])
        facility_rating = st.slider("Facility Rating", min_value=1, max_value=5, value=3)
        exam_difficulty = st.selectbox("Exam Difficulty", ["Easy", "Medium", "Hard"])
        
    st.markdown('</div>', unsafe_allow_html=True)
    submit_val = st.form_submit_button("Predict Result")

if submit_val:
    if model is None:
        st.error("Model is not loaded properly. Cannot run prediction.")
    else:
        # Categorical encoding maps (Adjust these mappings if your training used different values/encoders)
        gender_map = {"Male": 0, "Female": 1, "Other": 2}
        internet_map = {"No": 0, "Yes": 1}
        sleep_quality_map = {"Poor": 0, "Average": 1, "Good": 2, "Excellent": 3}
        study_method_map = {"Self-Study": 0, "Online": 1, "Group": 2, "Hybrid": 3}
        exam_difficulty_map = {"Easy": 0, "Medium": 1, "Hard": 2}
        
        # Simple placeholder mapping for course text input (modify if using a saved LabelEncoder)
        course_encoded = hash(course) % 100

        # Construct input dataframe matching the expected features
        input_data = pd.DataFrame([[
            age,
            gender_map.get(gender, 0),
            course_encoded,
            study_hours,
            class_attendance,
            internet_map.get(internet_access, 0),
            sleep_hours,
            sleep_quality_map.get(sleep_quality, 0),
            study_method_map.get(study_method, 0),
            facility_rating,
            exam_difficulty_map.get(exam_difficulty, 0)
        ]], columns=[
            'age', 'gender', 'course', 'study_hours', 'class_attendance', 
            'internet_access', 'sleep_hours', 'sleep_quality', 
            'study_method', 'facility_rating', 'exam_difficulty'
        ])

        try:
            prediction = model.predict(input_data)
            st.success(f"Predicted Performance Score: **{prediction[0]:.2f}**")
        except Exception as e:
            st.error(f"Prediction error: {e}. Please ensure feature columns and formatting match your training pipeline.")
