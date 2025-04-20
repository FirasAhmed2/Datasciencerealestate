import streamlit as st
import pandas as pd
import joblib

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
        # Load the trained pipeline
        model = joblib.load('finalGBDTmodel.joblib')
        
        # Load the encoders
        le_building = joblib.load('label_encoder_building.joblib')
        le_borough = joblib.load('label_encoder_borough.joblib')
        
        st.success("All artifacts loaded successfully!")
        return model, le_building, le_borough
    except FileNotFoundError as e:
        st.error(f"File not found: {str(e)}")
        st.info("Please ensure these files exist in your deployment directory:")
        st.info("- finalGBDTmodel.joblib")
        st.info("- label_encoder_building.joblib")
        st.info("- label_encoder_borough.joblib")
        return None, None, None
    except Exception as e:
        st.error(f"Loading failed: {str(e)}")
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
        year_built = st.slider("Year Built", 1800, 2023, 1990)
        gross_sqft = st.number_input("Gross Square Feet", 500, 100000, 1500, step=100)
        land_sqft = st.number_input("Land Square Feet", 500, 100000, 2000, step=100)
        units = st.number_input("Residential Units", 1, 1000, 2)
        
        building_class_category = st.selectbox(
            "Building Class Category",
            options=le_building.classes_
        )
        
        borough = st.selectbox(
            "Borough (1=Manhattan, 2=Brooklyn, 3=Queens, 4=Bronx, 5=Staten Island)",
            options=le_borough.classes_
        )
    
    # Encode the BOROUGH and BUILDING CLASS CATEGORY features
    try:
        borough_encoded = le_borough.transform([borough])[0]
        building_class_encoded = le_building.transform([building_class_category])[0]
    except ValueError as e:
        st.error(f"Encoding error: {str(e)}. Ensure the input values match the training data.")
        st.stop()
    
    # Prepare input data with encoded features
    input_df = pd.DataFrame({
        'GROSS SQUARE FEET': [gross_sqft],
        'LAND SQUARE FEET': [land_sqft],
        'YEAR BUILT': [year_built],
        'RESIDENTIAL UNITS': [units],
        'BUILDING_CLASS_ENCODED': [building_class_encoded],  # Use the encoded value
        'BOROUGH_ENCODED': [borough_encoded]  # Use the encoded value
    })
    
    # Ensure column order matches training data
    input_df = input_df[[
        'GROSS SQUARE FEET', 
        'LAND SQUARE FEET', 
        'YEAR BUILT', 
        'RESIDENTIAL UNITS', 
        'BUILDING_CLASS_ENCODED', 
        'BOROUGH_ENCODED'
    ]]
    
    if st.button("Predict Price Class", type="primary"):
        with st.spinner('Making prediction...'):
            try:
                # Pass the input data through the pipeline
                prediction = model.predict(input_df)
                proba = model.predict_proba(input_df)[0]
                
                # Display results
                st.subheader("Prediction Result")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Predicted Class", 
                             "High Price" if prediction[0] == 1 else "Low Price")
                
                with col2:
                    st.metric("Confidence", 
                             f"{max(proba)*100:.1f}%")
                
                # Feature importance visualization
                st.subheader("Feature Importance")
                xgb_model = model.named_steps['xgb']
                importance = xgb_model.feature_importances_
                
                # Create readable feature names
                feature_names = [
                    'GROSS SQUARE FEET',
                    'LAND SQUARE FEET',
                    'YEAR BUILT',
                    'RESIDENTIAL UNITS',
                    'BUILDING CLASS',
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
