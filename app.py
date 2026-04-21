
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('best_model.pkl')

# Define mappings for categorical features (based on your LabelEncoder training)
gender_mapping = {'Male': 1, 'Female': 0}
education_mapping = {"Bachelor's Degree": 0, "Master's Degree": 1, "PhD": 2}

# For 'Job Title', a full mapping from the original dataset would be ideal.
# For this demo, we'll ask for the encoded numerical value directly, 
# or you can expand this to include a dropdown with all 174 original job titles and their encoded values.
# Example encoded job titles from your training (replace with actual values from your le.classes_ if available):
job_title_examples = {
    'Software Engineer': 159,
    'Data Analyst': 17,
    'Senior Manager': 130,
    'Sales Associate': 101,
    'Director': 22
}

st.title('Salary Prediction App')
st.write('Enter employee details to predict their salary.')

# Input fields for features
age = st.slider('Age', 18, 65, 30)
gender = st.selectbox('Gender', list(gender_mapping.keys()))
education_level = st.selectbox('Education Level', list(education_mapping.keys()))

# For Job Title, we'll provide a selectbox with examples or allow direct numerical input
job_title_option = st.radio("How would you like to enter Job Title?", ('Select from examples', 'Enter encoded number'))
encoded_job_title = 0 # Default value

if job_title_option == 'Select from examples':
    selected_job_title_name = st.selectbox('Job Title (select example)', list(job_title_examples.keys()))
    encoded_job_title = job_title_examples[selected_job_title_name]
    st.write(f"Encoded Job Title: {encoded_job_title}")
else:
    encoded_job_title = st.number_input('Job Title (enter encoded number)', min_value=0, max_value=173, value=159)

years_of_experience = st.slider('Years of Experience', 0.0, 30.0, 5.0, 0.5)

if st.button('Predict Salary'):
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame([[age,
                                  gender_mapping[gender],
                                  education_mapping[education_level],
                                  encoded_job_title,
                                  years_of_experience]],
                                columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f'Predicted Salary: ${prediction:,.2f}')
