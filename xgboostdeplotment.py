import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import DistanceMetric

# Register distance metric
DistanceMetric.get_metric('euclidean')

# Page configuration
st.set_page_config(
    page_title="NYC Property Class Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('finalGBDTmodel.joblib')
        le_building = joblib.load('label_encoder_building.joblib')
        le_borough = joblib.load('label_encoder_borough.joblib')
        st.success("All artifacts loaded successfully!")
        return model, le_building, le_borough
    except Exception as e:
        st.error(f"Loading failed: {str(e)}")
        st.info("Please ensure these files exist:")
        st.info("- finalGBDTmodel.joblib")
        st.info("- label_encoder_building.joblib")
        st.info("- label_encoder_borough.joblib")
        return None, None, None

def main():
    st.title("🏙️ NYC Property Class Predictor (XGBoost)")
    st.markdown("Predict whether a property will be above or below median price")
    
    model, le_building, le_borough = load_artifacts()
    if None in [model, le_building, le_borough]:
        st.stop()
    
    with st.sidebar:
        st.header("Property Details")
        gross_sqft = st.number_input("GROSS SQUARE FEET", 500, 100000, 1500)
        land_sqft = st.number_input("LAND SQUARE FEET", 500, 100000, 2000)
        year_built = st.number_input("YEAR BUILT", 1800, 2023, 1990)
        units = st.number_input("RESIDENTIAL UNITS", 1, 1000, 2)
        
        building_class = st.selectbox(
            "BUILDING CLASS CATEGORY",
            options=le_building.classes_
        )
        
        # Fixed borough selection to show both names and codes
        borough_options = list(zip(le_borough.classes_, range(1, 6)))
        selected_borough = st.selectbox(
            "BOROUGH (1=Manhattan, 2=Brooklyn, 3=Queens, 4=Bronx, 5=Staten Island)",
            options=borough_options,
            format_func=lambda x: f"{x[1]}={x[0]}"
        )
        borough = selected_borough[1]  # Get the integer code
    
    if st.button("Predict Price Class"):
        with st.spinner('Processing...'):
            try:
                # Prepare input data
                input_df = pd.DataFrame({
                    'GROSS SQUARE FEET': [gross_sqft],
                    'LAND SQUARE FEET': [land_sqft],
                    'YEAR BUILT': [year_built],
                    'RESIDENTIAL UNITS': [units],
                    'BUILDING CLASS CATEGORY': [building_class],
                    'BOROUGH': [borough]  # Using the integer code directly
                })
                
                # Encode categorical variables
                input_df['BUILDING_CLASS_ENCODED'] = le_building.transform(
                    input_df['BUILDING CLASS CATEGORY']
                )
                # Borough is already encoded as integer
                input_df['BOROUGH_ENCODED'] = input_df['BOROUGH']  
                
                # Select features for prediction
                X = input_df[[
                    'GROSS SQUARE FEET',
                    'LAND SQUARE FEET',
                    'YEAR BUILT',
                    'RESIDENTIAL UNITS',
                    'BUILDING_CLASS_ENCODED',
                    'BOROUGH_ENCODED'
                ]]
                
                # Make prediction
                prediction = model.predict(X)
                proba = model.predict_proba(X)[0]
                
                # Display results
                st.subheader("Prediction Result")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Class", 
                            "High Price" if prediction[0] == 1 else "Low Price")
                with col2:
                    st.metric("Confidence", f"{max(proba)*100:.1f}%")
                
                # Feature importance
                if hasattr(model, 'named_steps'):
                    xgb_model = model.named_steps.get('xgb')
                    if xgb_model is not None:
                        st.subheader("Feature Importance")
                        importance = xgb_model.feature_importances_
                        feature_names = [
                            'GROSS_SQ_FT',
                            'LAND_SQ_FT',
                            'YEAR_BUILT',
                            'RESIDENTIAL_UNITS',
                            'BUILDING_CLASS',
                            'BOROUGH'
                        ]
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importance
                        }).sort_values('Importance', ascending=False)
                        st.bar_chart(importance_df.set_index('Feature'))
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.info("""
                Common issues:
                - Verify all fields are filled correctly
                - Check borough selection matches training data
                - Ensure model files are in the correct location
                """)

if __name__ == "__main__":
    main()
