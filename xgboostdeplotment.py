import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys
from importlib import reload

# Fix for EuclideanDistance error in cloud environments
def fix_sklearn_loading():
    try:
        import sklearn.metrics._dist_metrics
        reload(sklearn.metrics._dist_metrics)
        
        # Create dummy EuclideanDistance class if needed
        if not hasattr(sklearn.metrics._dist_metrics, 'EuclideanDistance'):
            from sklearn.metrics import DistanceMetric
            class EuclideanDistance:
                def __init__(self, **kwargs):
                    pass
                def pairwise(self, X, Y=None):
                    return DistanceMetric.get_metric('euclidean').pairwise(X, Y)
            sklearn.metrics._dist_metrics.EuclideanDistance = EuclideanDistance
    except Exception as e:
        st.warning(f"Sklearn compatibility fix partially failed: {str(e)}")

# Apply the fix before loading models
fix_sklearn_loading()

# Page configuration
st.set_page_config(
    page_title="NYC Property Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced model loader with multiple fallbacks
@st.cache_resource
def load_artifacts():
    def try_loading(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Standard loading failed for {path}: {str(e)}")
            try:
                # Try with pickle directly
                import pickle
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                st.error(f"Alternative loading failed for {path}: {str(e)}")
                return None

    try:
        model = try_loading('finalGBDTmodel.joblib')
        le_building = try_loading('label_encoder_building.joblib')
        le_borough = try_loading('label_encoder_borough.joblib')
        
        if None in [model, le_building, le_borough]:
            return None, None, None
            
        st.success("All artifacts loaded successfully!")
        return model, le_building, le_borough
        
    except Exception as e:
        st.error(f"Artifact loading failed: {str(e)}")
        return None, None, None

def main():
    st.title("🏙️ NYC Property Class Predictor")
    
    model, le_building, le_borough = load_artifacts()
    if None in [model, le_building, le_borough]:
        st.stop()
    
    with st.form("prediction_form"):
        st.header("Property Details")
        gross_sqft = st.number_input("GROSS SQUARE FEET", 500, 100000, 1500)
        land_sqft = st.number_input("LAND SQUARE FEET", 500, 100000, 2000)
        year_built = st.number_input("YEAR BUILT", 1800, 2023, 1990)
        units = st.number_input("RESIDENTIAL UNITS", 1, 1000, 2)
        
        building_class = st.selectbox(
            "BUILDING CLASS CATEGORY",
            options=le_building.classes_
        )
        
        borough_name = st.selectbox(
            "BOROUGH",
            options=le_borough.classes_
        )
        
        submitted = st.form_submit_button("Predict Price Class")
    
    if submitted:
        with st.spinner('Processing prediction...'):
            try:
                # Prepare input data
                input_df = pd.DataFrame({
                    'GROSS SQUARE FEET': [float(gross_sqft)],
                    'LAND SQUARE FEET': [float(land_sqft)],
                    'YEAR BUILT': [int(year_built)],
                    'RESIDENTIAL UNITS': [int(units)],
                    'BUILDING CLASS CATEGORY': [str(building_class)],
                    'BOROUGH': [str(borough_name)]
                })
                
                # Apply transformations
                input_df['BUILDING_CLASS_ENCODED'] = le_building.transform(
                    input_df['BUILDING CLASS CATEGORY']
                )
                input_df['BOROUGH_ENCODED'] = le_borough.transform(
                    input_df['BOROUGH']
                )
                
                # Select final features
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
                
                # Display results
                st.subheader("Prediction Result")
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]
                    confidence = f"{max(proba)*100:.1f}%"
                else:
                    confidence = "N/A"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Class", 
                             "High Price" if prediction[0] == 1 else "Low Price")
                with col2:
                    st.metric("Confidence", confidence)
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    try:
                        st.subheader("Feature Importance")
                        importance_df = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        st.bar_chart(importance_df.set_index('Feature'))
                    except Exception as e:
                        st.warning(f"Couldn't display feature importance: {str(e)}")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("""
                Troubleshooting:
                1. Verify all inputs are valid
                2. Check cloud logs for detailed errors
                3. Ensure model files are properly uploaded
                """)

if __name__ == "__main__":
    main()
