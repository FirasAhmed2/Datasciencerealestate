import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import DistanceMetric
class EuclideanDistance:
    def __init__(self, **kwargs):
        pass
    def pairwise(self, X, Y=None):
        return DistanceMetric.get_metric('euclidean').pairwise(X, Y)

# Monkey patch the module
import sklearn.metrics._dist_metrics
sklearn.metrics._dist_metrics.EuclideanDistance = EuclideanDistance

# Page configuration
st.set_page_config(
    page_title="NYC Property Class Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model and encoders loading
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
        st.info("Please make sure these files exist in your directory:")
        st.info("- finalGBDTmodel.joblib")
        st.info("- label_encoder_building.joblib")
        st.info("- label_encoder_borough.joblib")
        return None, None, None

def main():
    st.title("🏙️ NYC Property Class Predictor (XGBoost)")
    st.markdown("Predict whether a property will be above or below median price")
    
    # Load artifacts
    model, le_building, le_borough = load_artifacts()
    if None in [model, le_building, le_borough]:
        st.stop()
    
    # Input sidebar
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
        
        borough = st.selectbox(
            "BOROUGH 1=manhattan, 2=brooklyn, 3=queens, 4=bronx, 5=staten island",
            options=le_borough.classes_
        )
    
    # Prepare input data with EXACT column names used in training
    input_df = pd.DataFrame({
        'GROSS SQUARE FEET': [gross_sqft],
        'LAND SQUARE FEET': [land_sqft],
        'YEAR BUILT': [year_built],
        'RESIDENTIAL UNITS': [units],
        'BUILDING CLASS CATEGORY': [building_class],
        'BOROUGH': [borough]
    }, index=[0])
    
    if st.button("Predict Price Class", type="primary"):
        with st.spinner('Processing...'):
            try:
                # 1. Encode categorical variables exactly as in training
                input_df['BUILDING_CLASS_ENCODED'] = le_building.transform(
                    input_df['BUILDING CLASS CATEGORY'].astype(str)
                )
                input_df['BOROUGH_ENCODED'] = le_borough.transform(
                    input_df['BOROUGH'].astype(str)
                )
                
                # 2. Select only the features used in training (after encoding)
                features_for_prediction = [
                    'GROSS SQUARE FEET',
                    'LAND SQUARE FEET',
                    'YEAR BUILT',
                    'RESIDENTIAL UNITS',
                    'BUILDING_CLASS_ENCODED',
                    'BOROUGH_ENCODED'
                ]
                X = input_df[features_for_prediction]
                
                # 3. Make prediction (pipeline will handle imputation and SMOTE)
                prediction = model.predict(X)
                proba = model.predict_proba(X)[0]
                
                # Display results
                st.subheader("Prediction Result")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Predicted Class", 
                             "High Price" if prediction[0] == 1 else "Low Price")
                
                with col2:
                    st.metric("Confidence", 
                             f"{max(proba)*100:.1f}%")
                
                # Feature importance
                st.subheader("Feature Importance")
                xgb_model = model.named_steps['xgb']
                importance = xgb_model.feature_importances_
                
                # Create readable feature names
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
                - Make sure all fields are filled
                - Building class and borough must match training options
                - Check the format of your input values
                """)

if __name__ == "__main__":
    main()
