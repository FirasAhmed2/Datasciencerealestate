import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Custom loader to handle scikit-learn version differences
def safe_load_model(path):
    try:
        # First try normal loading
        return joblib.load(path)
    except Exception as e:
        st.warning(f"First load attempt failed: {str(e)}")
        try:
            # Try with custom encodings
            import sklearn
            sklearn.set_config(enable_metadata_routing=False)
            model = joblib.load(path)
            st.success("Loaded with compatibility settings")
            return model
        except Exception as e:
            st.error(f"Final load failed: {str(e)}")
            return None

# Page configuration
st.set_page_config(
    page_title="NYC Property Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_artifacts():
    try:
        model = safe_load_model('finalGBDTmodel.joblib')
        if model is None:
            return None, None, None
            
        le_building = joblib.load('label_encoder_building.joblib')
        le_borough = joblib.load('label_encoder_borough.joblib')
        
        # Verify borough encoding matches our expected 1-5 scheme
        test_values = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
        encoded = le_borough.transform(test_values)
        if not all(encoded == np.array([1, 2, 3, 4, 5])):
            st.warning("Borough encoding doesn't match expected 1-5 scheme")
        
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

    # Input form
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
                # Create input DataFrame with correct types
                input_data = {
                    'GROSS SQUARE FEET': float(gross_sqft),
                    'LAND SQUARE FEET': float(land_sqft),
                    'YEAR BUILT': int(year_built),
                    'RESIDENTIAL UNITS': int(units),
                    'BUILDING CLASS CATEGORY': str(building_class),
                    'BOROUGH': str(borough_name)
                }
                
                # Convert to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Apply same transformations as training
                input_df['BUILDING_CLASS_ENCODED'] = le_building.transform(
                    input_df['BUILDING CLASS CATEGORY']
                )
                input_df['BOROUGH_ENCODED'] = le_borough.transform(
                    input_df['BOROUGH']
                )
                
                # Prepare final feature set
                features = [
                    'GROSS SQUARE FEET',
                    'LAND SQUARE FEET',
                    'YEAR BUILT',
                    'RESIDENTIAL UNITS',
                    'BUILDING_CLASS_ENCODED',
                    'BOROUGH_ENCODED'
                ]
                X = input_df[features]
                
                # Make prediction
                prediction = model.predict(X)
                proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else [0, 0]
                
                # Display results
                st.subheader("Prediction Result")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Class", 
                            "High Price" if prediction[0] == 1 else "Low Price")
                with col2:
                    st.metric("Confidence", 
                             f"{max(proba)*100:.1f}%" if hasattr(model, "predict_proba") else "N/A")
                
                # Show feature importance if available
                if hasattr(model, 'feature_importances_'):
                    try:
                        st.subheader("Feature Importance")
                        importance_df = pd.DataFrame({
                            'Feature': features,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        st.bar_chart(importance_df.set_index('Feature'))
                    except Exception as e:
                        st.warning(f"Couldn't display feature importance: {str(e)}")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("""
                Troubleshooting steps:
                1. Verify all input values are valid
                2. Check model files haven't been corrupted
                3. Ensure you're using compatible Python packages
                """)

if __name__ == "__main__":
    main()
