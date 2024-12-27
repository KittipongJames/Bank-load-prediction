import streamlit as st
import joblib
import numpy as np


#load the model
model = joblib.load('clf_model.pkl')

def loan_prediction(inputs):
    # Change input data to numpy array and reshape
    input_as_np_array = np.array(inputs).reshape(1, -1)

    prediction = model.predict(input_as_np_array)

    if prediction[0] == 0:
        return 'This person should not get a loan'
    else:
        return 'This person is qualified for a loan'
    
def main():
    st.title('Loan Status Application App')

    # Getting the input from the user
    gender = st.text_input('gender', placeholder='Male:1, Female:0')
    married = st.text_input('married', placeholder='No:0, Yes:1')
    dependents = st.text_input('dependents')
    education = st.text_input('education', placeholder='Graduate:1, Not Graduate:0')
    self_employed = st.text_input('self_employed', placeholder='No:0, Yes:1')
    applicant_income = st.text_input('applicant_income')
    co_applicant_income = st.text_input('co_applicant_income')
    loan_amount = st.text_input('loan_amount')
    loan_amount_term = st.text_input('loan_amount_term')
    credit_history = st.text_input('credit_history')
    property_area = st.text_input('property_area', placeholder='rural:0, semiurban:1, urban:2')

    # Code for prediction
    pred = ''

    # Create a button for the prediction
    if st.button('Check if this person qualifies'):
        pred = loan_prediction([gender, married, dependents, education, self_employed, applicant_income, co_applicant_income, loan_amount, loan_amount_term, credit_history, property_area])
        st.success(pred)

if __name__ == '__main__':
    main()
