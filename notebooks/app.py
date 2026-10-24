import streamlit as st
import pandas as pd
import joblib

# 1. Load the trained XGBoost Pipeline
@st.cache_resource
def load_model():
    # Adjust path if your app.py is located elsewhere
    return joblib.load("../src/models/xgboost_pipeline.pkl")

pipeline = load_model()

# 2. Set up the UI Headers
st.set_page_config(page_title="C-Section Risk Predictor", layout="centered")
st.title("Cesarean Section Risk Stratification")
st.write("Enter patient demographics and institutional details based on BDHS 2022 indicators to generate a predictive risk score.")
st.divider()

# 3. Create Input Fields in a 2-Column Layout
col1, col2 = st.columns(2)

with col1:
    mothers_age = st.number_input("Maternal Age (years)", min_value=15, max_value=50, value=25)
    education = st.selectbox("Highest Education Level", ["No Education", "Primary", "Secondary", "Higher"])
    wealth_index = st.selectbox("Household Wealth Quintile", ["Poorest", "Poorer", "Middle", "Richer", "Richest"])
    residence = st.selectbox("Residence Type", ["Rural", "Urban"])

with col2:
    region = st.selectbox("Geographic Region", ["Barishal", "Chattogram", "Dhaka", "Khulna", "Mymensingh", "Rajshahi", "Rangpur", "Sylhet"])
    anc_visits = st.number_input("Total ANC Visits", min_value=0, max_value=40, value=4)
    delivery_facility = st.selectbox("Expected Delivery Facility", ["Home", "Private Facility", "Public Facility", "Other/Unknown"])
    birth_order = st.number_input("Birth Order", min_value=1, max_value=15, value=1)

st.divider()

# 4. Prediction Logic
if st.button("Calculate Surgical Probability", type="primary"):
    
    # 5. Map UI Inputs to the EXACT One-Hot Encoded DataFrame format from pd.get_dummies
    # 5. Map UI Inputs to the EXACT One-Hot Encoded DataFrame format from pd.get_dummies
    input_dict = {
        'mothers_age': mothers_age,
        'anc_visits': float(anc_visits),
        'birth_order': birth_order,
        
        # Education (Base: No Education)
        'education_primary': 1 if education == 'Primary' else 0,
        'education_secondary': 1 if education == 'Secondary' else 0,
        'education_higher': 1 if education == 'Higher' else 0,
        
        # Wealth Index (Base: Poorest)
        'wealth_index_poorer': 1 if wealth_index == 'Poorer' else 0,
        'wealth_index_middle': 1 if wealth_index == 'Middle' else 0,
        'wealth_index_richer': 1 if wealth_index == 'Richer' else 0,
        'wealth_index_richest': 1 if wealth_index == 'Richest' else 0,
        
        # Residence (Base: Urban)  <--- FIXED HERE
        'residence_type_rural': 1 if residence == 'Rural' else 0,
        
        # Region (Base: Barishal)
        'region_chattogram': 1 if region == 'Chattogram' else 0,
        'region_dhaka': 1 if region == 'Dhaka' else 0,
        'region_khulna': 1 if region == 'Khulna' else 0,
        'region_mymensingh': 1 if region == 'Mymensingh' else 0,
        'region_rajshahi': 1 if region == 'Rajshahi' else 0,
        'region_rangpur': 1 if region == 'Rangpur' else 0,
        'region_sylhet': 1 if region == 'Sylhet' else 0,
        
        # Delivery Facility (Base: Home)
        'delivery_facility_group_Other/Unknown': 1 if delivery_facility == 'Other/Unknown' else 0,
        'delivery_facility_group_Private Facility': 1 if delivery_facility == 'Private Facility' else 0,
        'delivery_facility_group_Public Facility': 1 if delivery_facility == 'Public Facility' else 0
    }
    
    # Ensure column order perfectly matches what the pipeline expects
    input_df = pd.DataFrame([input_dict])
    
    # 6. Generate Prediction
    # The pipeline handles the median imputation for NaNs and the Standard Scaling automatically
    probability = pipeline.predict_proba(input_df)[0][1] * 100
    prediction = pipeline.predict(input_df)[0]
    
    # 7. Display Results
    if prediction == 1:
        st.error(f"### High Probability of Cesarean Section ({probability:.1f}%)")
        st.write("The algorithmic baseline suggests strong socioeconomic and institutional risk factors associated with surgical delivery.")
    else:
        st.success(f"### High Probability of Vaginal Delivery ({100 - probability:.1f}%)")
        st.write("The algorithmic baseline aligns with standard vaginal delivery outcomes.")