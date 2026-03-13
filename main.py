import streamlit as st
import pandas as pd
from prediction_helper import predict

st.title("NBFC Credit Risk Prediction Tool")

st.subheader("Applicant Information")

# ----------- LAYOUT -----------

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)

# ----------- ROW 1 -----------

with row1[0]:
    age = st.number_input("Age", min_value=18, max_value=80, value=35)

with row1[1]:
    income = st.number_input("Income (Annual)", min_value=0, value=1200000)

with row1[2]:
    loan_amount = st.number_input("Loan Amount", min_value=0, value=500000)

# ----------- ROW 2 -----------

with row2[0]:
    loan_tenure = st.number_input("Loan Tenure (months)", min_value=1, value=36)

with row2[1]:
    delinquent_months = st.number_input("Delinquent Months", min_value=0, value=0)

with row2[2]:
    total_dpd = st.number_input("Total DPD", min_value=0, value=0)

# ----------- ROW 3 -----------

with row3[0]:
    credit_utilization_ratio = st.number_input(
        "Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=30.0
    )

with row3[1]:
    num_open_accounts = st.number_input("Open Loan Accounts", min_value=0, value=1)

with row3[2]:
    residence_type = st.selectbox(
        "Residence Type",
        ["Owned", "Rented","Mortgage"]
    )

# ----------- ROW 4 -----------

with row4[0]:
    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Home", "Auto", "Education", "Personal"]
    )

with row4[1]:
    loan_type = st.selectbox(
        "Loan Type",
        ["Secured", "Unsecured"]
    )

# Empty column just to maintain layout symmetry
with row4[2]:
    st.write("")

# ----------- DERIVED VARIABLES -----------

loan_to_income_ratio = loan_amount / income if income > 0 else 0
delinquency_ratio = delinquent_months / loan_tenure if loan_tenure > 0 else 0
avg_dpd = total_dpd / delinquent_months if delinquent_months > 0 else 0

st.subheader("Derived Risk Metrics")

st.write(f"Loan to Income Ratio: **{loan_to_income_ratio:.2f}**")
st.write(f"Delinquency Ratio: **{delinquency_ratio:.2f}**")
st.write(f"Average DPD: **{avg_dpd:.2f}**")

# ----------- PREDICTION BUTTON -----------

if st.button("Calculate Risk"):
    st.subheader("Input Data for Model")


    probability,credit_score,rating = predict(age,income,loan_amount,loan_tenure,delinquency_ratio,
            avg_dpd,credit_utilization_ratio,num_open_accounts,residence_type,loan_purpose,
            loan_type)

    st.write(f'Default Probability : {probability}')
    st.write(f'Credit Risk Score : {credit_score}')
    st.write(f'Rating : {rating}')


    st.success("Prediction will appear here")
