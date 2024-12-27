import streamlit as st
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # or whatever model type you used

# Load the model
try:
    model = joblib.load('clf_model.pkl')
except FileNotFoundError:
    st.error("Model file 'clf_model.pkl' not found. Please ensure it's in the same directory as this script.")
    st.stop()

def loan_prediction(inputs):
    try:
        # Convert all inputs to float
        inputs = [float(x) for x in inputs]
        input_as_np_array = np.array(inputs).reshape(1, -1)
        prediction = model.predict(input_as_np_array)
        
        return 'This person is qualified for a loan' if prediction[0] == 1 else 'This person should not get a loan'
    except ValueError:
        return "Error: Please ensure all fields contain valid numbers"

def main():
    st.title('Loan Status Application App')
    
    # Adding descriptions and input validation
    gender = st.selectbox('Gender', options=['0', '1'], format_func=lambda x: 'Female' if x == '0' else 'Male')
    married = st.selectbox('Married', options=['0', '1'], format_func=lambda x: 'No' if x == '0' else 'Yes')
    dependents = st.number_input('Number of Dependents', min_value=0, max_value=10, value=0)
    education = st.selectbox('Education', options=['0', '1'], format_func=lambda x: 'Not Graduate' if x == '0' else 'Graduate')
    self_employed = st.selectbox('Self Employed', options=['0', '1'], format_func=lambda x: 'No' if x == '0' else 'Yes')
    applicant_income = st.number_input('Applicant Income', min_value=0)
    co_applicant_income = st.number_input('Co-applicant Income', min_value=0)
    loan_amount = st.number_input('Loan Amount', min_value=0)
    loan_amount_term = st.number_input('Loan Amount Term (in months)', min_value=0)
    credit_history = st.selectbox('Credit History', options=['0', '1'], format_func=lambda x: 'No' if x == '0' else 'Yes')
    property_area = st.selectbox('Property Area', 
                               options=['0', '1', '2'], 
                               format_func=lambda x: 'Rural' if x == '0' else ('Semi-Urban' if x == '1' else 'Urban'))

    # Create a button for prediction
    if st.button('Check if this person qualifies'):
        inputs = [gender, married, dependents, education, self_employed, 
                 applicant_income, co_applicant_income, loan_amount, 
                 loan_amount_term, credit_history, property_area]
        
        # Check if all fields are filled
        if all(str(x).strip() for x in inputs):
            pred = loan_prediction(inputs)
            if "Error" in pred:
                st.error(pred)
            else:
                st.success(pred)
        else:
            st.error("Please fill in all fields")

if __name__ == '__main__':
    main()
