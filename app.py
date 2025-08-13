# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
from pathlib import Path
import os
import time

def ensure_arrow_compatibility(df):
    """Ensure DataFrame is compatible with Arrow serialization"""
    try:
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype('string')
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].dtype == 'float64':
                df_clean[col] = df_clean[col].astype('float32')
            elif df_clean[col].dtype == 'int64':
                df_clean[col] = df_clean[col].astype('int32')
        return df_clean
    except Exception as e:
        st.error(f"Error ensuring Arrow compatibility: {e}")
        return df

def validate_dataframe(df):
    """Validate DataFrame for Streamlit compatibility"""
    try:
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            for col in object_cols:
                try:
                    df[col] = df[col].astype('string')
                except Exception as e:
                    return False
        
        inf_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
        
        if inf_cols:
            for col in inf_cols:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].fillna(df[col].median())
        
        string_cols = df.select_dtypes(include=['string']).columns
        for col in string_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna('Unknown')
        
        return True
    except Exception as e:
        return False

# Page configuration
st.set_page_config(
    page_title="♻️ Waste Management Analytics & Predictor",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        background: linear-gradient(90deg, #4CAF50, #45a049, #2E7D32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-radius: 15px;
        padding: 1.5rem;
        border-left: 5px solid #2E7D32;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        color: black;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
    }
    
    /* Form styling */
    .stForm {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
    
    /* Form labels styling */
    .stForm label {
        color: black !important;
        font-weight: 600;
    }
    
    /* Headers styling */
    .stForm h3, .stForm h4 {
        color: black !important;
        font-weight: 700;
    }
    
    /* Prediction result styling */
    .prediction-result {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 2px solid #4CAF50;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.2);
    }
    
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        margin: 1rem 0;
    }
    
    /* Prediction result text styling */
    .prediction-result h3, .prediction-result p, .prediction-result div {
        color: black !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4CAF50, #45a049);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)


# Load data and model
@st.cache_data
def load_data():
    """Load and cache the waste management data"""
    try:
        # Define a list of possible paths for the data file
        possible_paths = [
            Path.cwd() / "waste_data.csv",
            Path("waste_data.csv")
        ]
        
        data_path = None
        for path in possible_paths:
            if path.exists():
                data_path = path
                break
        
        if data_path is None:
            st.error(f"Data file not found. Tried paths: {[str(p) for p in possible_paths]}")
            return None
        
        df = pd.read_csv(data_path)
        
        # Check for Landfill Location column and split if necessary
        if 'Landfill Location (Lat, Long)' in df.columns:
            # Check if columns are already split
            if 'Landfill_Lat' not in df.columns or 'Landfill_Long' not in df.columns:
                try:
                    df[['Landfill_Lat', 'Landfill_Long']] = df['Landfill Location (Lat, Long)'].str.strip().str.replace('"', '').str.split(', ', expand=True).astype(float)
                except Exception as e:
                    st.warning(f"Could not parse 'Landfill Location (Lat, Long)' column. Map functionality may not work correctly. Error: {e}")
                    df['Landfill_Lat'] = np.nan
                    df['Landfill_Long'] = np.nan
        else:
            st.warning("'Landfill Location (Lat, Long)' column not found. Map functionality will be disabled.")
            df['Landfill_Lat'] = np.nan
            df['Landfill_Long'] = np.nan
            
        # Ensure all necessary columns exist for the model to work
        required_cols = ['City/District', 'Waste Type', 'Disposal Method', 'Waste Generated (Tons/Day)', 
                         'Population Density (People/km²)', 'Municipal Efficiency Score (1-10)', 
                         'Cost of Waste Management (₹/Ton)', 'Awareness Campaigns Count', 
                         'Landfill Capacity (Tons)', 'Year']
        
        for col in required_cols:
            if col not in df.columns:
                st.warning(f"Required column '{col}' not found. Adding with placeholder values.")
                if df.empty:
                    # If DataFrame is empty, create a placeholder column with a single value
                    df[col] = ['Placeholder'] if df[col].dtype == 'object' else [0]
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if df[col].dtype == 'object' else df[col].median())
        
        df = ensure_arrow_compatibility(df)
        return df
    except Exception as e:
        st.error(f"Unexpected error loading data: {e}")
        return None

@st.cache_resource
def load_model():
    """Load and cache the trained model"""
    try:
        # Define a list of possible paths for the model file
        possible_paths = [
            Path.cwd() / "catboost_tuned_model.pkl",
            Path("catboost_tuned_model.pkl")
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            st.error(f"Model file 'catboost_tuned_model.pkl' not found.")
            return None
            
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the data and model
df = load_data()
model = load_model()

# Check if data or model failed to load
if df is None or model is None:
    st.stop()

# Load or create the predictions DataFrame
predictions_file = "predictions.csv"
if os.path.exists(predictions_file):
    predictions_df = pd.read_csv(predictions_file)
else:
    predictions_df = pd.DataFrame(columns=['City_District', 'Waste_Type', 'Disposal_Method', 
                                           'Waste_Generated_Tons_Per_Day', 'Population_Density_People_Per_km2',
                                           'Municipal_Efficiency_Score_1_10', 'Cost_of_Waste_Management_Rs_Per_Ton',
                                           'Awareness_Campaigns_Count', 'Landfill_Capacity_Tons', 'Year',
                                           'Recycling_Rate_Predicted'])

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Predictor"])

def predictor_page():
    """Renders the prediction UI"""
    st.markdown("<h1 class='main-title'>Recycling Rate Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Input parameters to predict the recycling rate for a city.</p>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown("<h3>Input Parameters</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            city = st.selectbox("🏙️ City/District", options=df['City/District'].unique())
            waste_type = st.selectbox("🗑️ Waste Type", options=df['Waste Type'].unique())
            disposal_method = st.selectbox("♻️ Disposal Method", options=df['Disposal Method'].unique())
            waste_generated = st.number_input("📈 Waste Generated (Tons/Day)", min_value=0.0, value=1000.0)
            pop_density = st.number_input("👥 Population Density (People/km²)", min_value=0, value=16000)

        with col2:
            efficiency_score = st.slider("⭐ Municipal Efficiency Score (1-10)", min_value=1, max_value=10, value=8)
            cost = st.number_input("💰 Cost of Waste Management (₹/Ton)", min_value=0.0, value=2000.0)
            campaigns = st.number_input("📣 Awareness Campaigns Count", min_value=0, value=20)
            landfill_capacity = st.number_input("🏞️ Landfill Capacity (Tons)", min_value=0.0, value=50000.0)
            year = st.number_input("📅 Year", min_value=2000, max_value=2100, value=2027)

        submit_button = st.form_submit_button("🔮 Predict Recycling Rate")

    if submit_button:
        # Create a DataFrame from the input data
        input_data = pd.DataFrame([{
            'City/District': city,
            'Waste Type': waste_type,
            'Disposal Method': disposal_method,
            'Waste Generated (Tons/Day)': waste_generated,
            'Population Density (People/km²)': pop_density,
            'Municipal Efficiency Score (1-10)': efficiency_score,
            'Cost of Waste Management (₹/Ton)': cost,
            'Awareness Campaigns Count': campaigns,
            'Landfill Capacity (Tons)': landfill_capacity,
            'Year': year
        }])
        
        # Define categorical features
        cat_features = ['City/District', 'Waste Type', 'Disposal Method']
        
        try:
            # Make prediction
            prediction = model.predict(input_data, cat_features=cat_features)[0]
            
            # Clamp the prediction to be between 0 and 100
            prediction = max(0, min(100, prediction))
            
            # Display the result
            st.markdown(f"""
                <div class="prediction-result">
                    <h3>Predicted Recycling Rate</h3>
                    <div class="prediction-value">{prediction:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Store the prediction
            new_prediction_row = pd.DataFrame([{
                'City_District': city,
                'Waste_Type': waste_type,
                'Disposal_Method': disposal_method,
                'Waste_Generated_Tons_Per_Day': waste_generated,
                'Population_Density_People_Per_km2': pop_density,
                'Municipal_Efficiency_Score_1_10': efficiency_score,
                'Cost_of_Waste_Management_Rs_Per_Ton': cost,
                'Awareness_Campaigns_Count': campaigns,
                'Landfill_Capacity_Tons': landfill_capacity,
                'Year': year,
                'Recycling_Rate_Predicted': prediction
            }])
            
            global predictions_df
            predictions_df = pd.concat([predictions_df, new_prediction_row], ignore_index=True)
            predictions_df.to_csv(predictions_file, index=False)
            
            st.success("Prediction has been saved to predictions.csv!")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

def dashboard_page():
    """Renders the dashboard UI with data visualizations"""
    st.markdown("<h1 class='main-title'>Waste Management Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Explore and analyze key waste management metrics.</p>", unsafe_allow_html=True)

    # Display key metrics
    if not df.empty:
        total_waste_generated = df['Waste Generated (Tons/Day)'].sum()
        avg_recycling_rate = df['Recycling Rate (%)'].mean()
        most_common_waste_type = df['Waste Type'].mode()[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-card'><h3>Total Waste Generated</h3><p>{total_waste_generated:,.0f} Tons/Day</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h3>Avg. Recycling Rate</h3><p>{avg_recycling_rate:.2f}%</p></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><h3>Most Common Waste</h3><p>{most_common_waste_type}</p></div>", unsafe_allow_html=True)

        # Plotly charts
        st.header("Waste Generated by Type")
        waste_by_type = df.groupby('Waste Type')['Waste Generated (Tons/Day)'].sum().reset_index()
        fig_pie = px.pie(waste_by_type, values='Waste Generated (Tons/Day)', names='Waste Type', 
                         title='Distribution of Waste Types', 
                         color_discrete_sequence=px.colors.qualitative.D3)
        st.plotly_chart(fig_pie, use_container_width=True)

        st.header("Recycling Rate over Years")
        avg_rate_by_year = df.groupby('Year')['Recycling Rate (%)'].mean().reset_index()
        fig_line = px.line(avg_rate_by_year, x='Year', y='Recycling Rate (%)', 
                           title='Average Recycling Rate by Year', markers=True)
        st.plotly_chart(fig_line, use_container_width=True)

        st.header("Waste Management Metrics by City")
        fig_bar = px.bar(df, x='City/District', y='Recycling Rate (%)', color='Disposal Method', 
                         title='Recycling Rate and Disposal Method by City', barmode='group')
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Interactive map
        st.header("Landfill Locations")
        map_df = df.dropna(subset=['Landfill_Lat', 'Landfill_Long']).drop_duplicates(subset=['Landfill Name'])
        
        if not map_df.empty:
            m = folium.Map(location=[map_df['Landfill_Lat'].mean(), map_df['Landfill_Long'].mean()], zoom_start=5)
            for _, row in map_df.iterrows():
                folium.Marker(
                    location=[row['Landfill_Lat'], row['Landfill_Long']],
                    tooltip=f"{row['Landfill Name']} - Capacity: {row['Landfill Capacity (Tons)']:,.0f} tons",
                    icon=folium.Icon(color='green', icon='trash', prefix='fa')
                ).add_to(m)
            folium_static(m, width=700)
        else:
            st.info("No landfill location data available to display on the map.")

    else:
        st.warning("No data available to display the dashboard.")

# Router
if page == "Predictor":
    predictor_page()
elif page == "Dashboard":
    dashboard_page()
