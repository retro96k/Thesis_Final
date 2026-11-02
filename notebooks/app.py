"""
Cesarean Section Risk Stratification Web Interface

This Streamlit application serves as the front-end deployment for the XGBoost 
machine learning pipeline trained on the BDHS 2022 dataset. It collects 
user-inputted sociodemographic and institutional data, manually structures 
the one-hot encoded feature matrix, and queries the serialized pipeline 
to output a real-time surgical delivery risk probability.
"""
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. Pipeline Instantiation & Memory Management
# ==========================================
@st.cache_resource
def load_model():
    """
    Loads the serialized scikit-learn pipeline (Imputer -> Scaler -> XGBoost).
    The @st.cache_resource decorator ensures the model is loaded into system memory 
    only once upon server initialization, drastically reducing inference latency 
    and preventing memory leaks during repeated user interactions.
    """
    # Adjust path if your app.py is executed from a different directory
    return joblib.load("../src/models/xgboost_pipeline.pkl")

pipeline = load_model()

# ==========================================
# 2. User Interface (UI) Configuration
# ==========================================
# Set the global configuration for the web page tab and layout structure
st.set_page_config(page_title="C-Section Risk Predictor", layout="centered")

# Render the primary application header and instructional text
st.title("Cesarean Section Risk Stratification")
st.write("Enter patient demographics and institutional details based on BDHS 2022 indicators to generate a predictive risk score.")
st.divider()

# ==========================================
# 3. Data Collection Module (Two-Column Grid)
# ==========================================
# Instantiate a dual-column layout to optimize screen real estate and user experience
col1, col2 = st.columns(2)

with col1:
    # Maternal Age (Continuous: Constrained between 15 and 50 to reflect standard reproductive age ranges)
    mothers_age = st.number_input("Maternal Age (years) [15-50]", min_value=15, max_value=50, value=25)
    
    # Education and Wealth (Categorical: Mapped strictly to standard DHS quintiles and levels)
    education = st.selectbox("Highest Education Level", ["No Education", "Primary", "Secondary", "Higher"])
    wealth_index = st.selectbox("Household Wealth Quintile", ["Poorest", "Poorer", "Middle", "Richer", "Richest"])
    
    # Residence (Categorical)
    residence = st.selectbox("Residence Type", ["Rural", "Urban"])

with col2:
    # Geography (Categorical: Divisional level classification)
    region = st.selectbox("Geographic Region", ["Barishal", "Chattogram", "Dhaka", "Khulna", "Mymensingh", "Rajshahi", "Rangpur", "Sylhet"])
    
    # Antenatal Care (Continuous: Constrained to prevent extreme outlier entries)
    anc_visits = st.number_input("Total ANC Visits [0-40]", min_value=0, max_value=40, value=4)
    
    # Facility (Categorical: The primary institutional catalyst evaluated in this study)
    delivery_facility = st.selectbox("Delivery Facility", ["Home", "Private Facility", "Public Facility", "Other/Unknown"])
    
    # Birth Order / Parity (Continuous)
    birth_order = st.number_input("Birth Order [1-15]", min_value=1, max_value=15, value=1)

st.divider()

# ==========================================
# 4. Inference Execution & Target Mapping
# ==========================================
# The block below executes only when the user clicks the primary calculation button
if st.button("Calculate Surgical Probability", type="primary"):
    
    # 5. Design Matrix Construction (Manual One-Hot Encoding)
    # The trained XGBoost pipeline expects exactly 21 features. Because the input 
    # is collected via raw strings in the UI, we must manually replicate the 
    # pd.get_dummies(drop_first=True) transformation applied during training.
    # We explicitly omit the "Base" categories to prevent multicollinearity errors.
    input_dict = {
        # Continuous Features
        'mothers_age': mothers_age,
        'anc_visits': float(anc_visits),
        'birth_order': birth_order,
        
        # Categorical: Education Level (Reference Category/Base: No Education)
        'education_primary': 1 if education == 'Primary' else 0,
        'education_secondary': 1 if education == 'Secondary' else 0,
        'education_higher': 1 if education == 'Higher' else 0,
        
        # Categorical: Household Wealth (Reference Category/Base: Poorest)
        'wealth_index_poorer': 1 if wealth_index == 'Poorer' else 0,
        'wealth_index_middle': 1 if wealth_index == 'Middle' else 0,
        'wealth_index_richer': 1 if wealth_index == 'Richer' else 0,
        'wealth_index_richest': 1 if wealth_index == 'Richest' else 0,
        
        # Categorical: Residence Type (Reference Category/Base: Urban)
        'residence_type_rural': 1 if residence == 'Rural' else 0,
        
        # Categorical: Geographic Region (Reference Category/Base: Barishal)
        'region_chattogram': 1 if region == 'Chattogram' else 0,
        'region_dhaka': 1 if region == 'Dhaka' else 0,
        'region_khulna': 1 if region == 'Khulna' else 0,
        'region_mymensingh': 1 if region == 'Mymensingh' else 0,
        'region_rajshahi': 1 if region == 'Rajshahi' else 0,
        'region_rangpur': 1 if region == 'Rangpur' else 0,
        'region_sylhet': 1 if region == 'Sylhet' else 0,
        
        # Categorical: Delivery Facility (Reference Category/Base: Home)
        'delivery_facility_group_Other/Unknown': 1 if delivery_facility == 'Other/Unknown' else 0,
        'delivery_facility_group_Private Facility': 1 if delivery_facility == 'Private Facility' else 0,
        'delivery_facility_group_Public Facility': 1 if delivery_facility == 'Public Facility' else 0
    }
    
    # Convert the processed dictionary into a single-row Pandas DataFrame for pipeline ingestion
    input_df = pd.DataFrame([input_dict])
    
    # ==========================================
    # 6. Model Prediction & Probability Extraction
    # ==========================================
    # The scikit-learn pipeline automatically applies the median imputation and 
    # StandardScaler to the numeric inputs. 'predict_proba' returns a 2D array; 
    # [0][1] strictly extracts the probability of the positive class (C-Section).
    probability = pipeline.predict_proba(input_df)[0][1] * 100
    prediction = pipeline.predict(input_df)[0]
    
    # ==========================================
    # 7. Dynamic Alert Rendering
    # ==========================================
    # Output color-coded administrative guidance based on the standard 50% classification threshold
    if prediction == 1:
        st.error(f"### High Probability of Cesarean Section ({probability:.1f}%)")
        st.write("The algorithmic baseline suggests strong socioeconomic and institutional risk factors associated with surgical delivery.")
    else:
        st.success(f"### High Probability of Vaginal Delivery ({100 - probability:.1f}%)")
        st.write("The algorithmic baseline aligns with standard vaginal delivery outcomes.")