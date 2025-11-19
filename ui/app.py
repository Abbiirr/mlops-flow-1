"""
NYC Taxi Trip Duration Prediction UI
User-friendly interface for model predictions
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime, time
import json

# Configuration
API_BASE_URL = "http://fastapi:8000"
PREDICTION_API_URL = "http://prediction-api:8080"

# Page configuration
st.set_page_config(
    page_title="NYC Taxi Trip Duration Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UX
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #145a8c;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        text-align: center;
        margin-top: 2rem;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if APIs are available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_model_info():
    """Get current champion model information"""
    try:
        response = requests.get(f"{PREDICTION_API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def get_test_samples():
    """Fetch test samples from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/data/test-samples?limit=100", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data['samples'])
    except Exception as e:
        st.error(f"Failed to fetch test samples: {str(e)}")
    return None


def predict_test_sample(sample_id):
    """Predict using a test sample"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/test",
            json={"sample_id": int(sample_id)},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Prediction failed: {response.text}")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
    return None


def predict_manual(features):
    """Predict with manual input"""
    try:
        response = requests.post(
            f"{PREDICTION_API_URL}/predict",
            json=features,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Prediction failed: {response.text}")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
    return None


def get_experiments():
    """Get list of experiments from MLflow"""
    try:
        response = requests.get(f"{API_BASE_URL}/experiments", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []


# Header
st.markdown('<h1 class="main-header">🚕 NYC Taxi Trip Duration Predictor</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ System Status")

    # Check API health
    if check_api_health():
        st.success("✅ API Connected")
    else:
        st.error("❌ API Unavailable")

    # Model information
    model_info = get_model_info()
    if model_info:
        st.subheader("📊 Current Model")
        st.info(f"**Version:** {model_info.get('model_version', 'N/A')}")

        if 'metrics' in model_info:
            metrics = model_info['metrics']
            st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
            st.metric("R² Score", f"{metrics.get('r2', 0):.4f}")

    st.markdown("---")

    # Navigation
    st.subheader("🧭 Navigation")
    page = st.radio(
        "Select Page:",
        ["🎯 Quick Predict", "🔍 Test Sample Prediction", "📈 Model Dashboard"],
        label_visibility="collapsed"
    )

# Main content based on page selection
if "Quick Predict" in page:
    st.header("🎯 Quick Prediction")
    st.write("Enter trip details below for instant duration prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Pickup Location")
        pickup_longitude = st.slider(
            "Longitude",
            min_value=-74.05,
            max_value=-73.75,
            value=-73.99,
            step=0.001,
            help="Pickup longitude (western NYC: -74.05, eastern: -73.75)"
        )
        pickup_latitude = st.slider(
            "Latitude",
            min_value=40.60,
            max_value=40.90,
            value=40.75,
            step=0.001,
            help="Pickup latitude (southern NYC: 40.60, northern: 40.90)"
        )

        st.subheader("📅 Pickup Time")
        pickup_date = st.date_input(
            "Date",
            value=datetime(2016, 3, 15),
            min_value=datetime(2016, 1, 1),
            max_value=datetime(2016, 6, 30)
        )
        pickup_time = st.time_input(
            "Time",
            value=time(14, 30)
        )

    with col2:
        st.subheader("🎯 Dropoff Location")
        dropoff_longitude = st.slider(
            "Longitude",
            min_value=-74.05,
            max_value=-73.75,
            value=-73.98,
            step=0.001,
            help="Dropoff longitude (western NYC: -74.05, eastern: -73.75)"
        )
        dropoff_latitude = st.slider(
            "Latitude",
            min_value=40.60,
            max_value=40.90,
            value=40.76,
            step=0.001,
            help="Dropoff latitude (southern NYC: 40.60, northern: 40.90)"
        )

        st.subheader("👥 Trip Details")
        passenger_count = st.select_slider(
            "Number of Passengers",
            options=[1, 2, 3, 4, 5, 6],
            value=1
        )

        vendor_id = st.selectbox(
            "Vendor",
            options=[1, 2],
            format_func=lambda x: f"Vendor {x}"
        )

    # Predict button
    if st.button("🚀 Predict Trip Duration", type="primary"):
        # Combine date and time
        pickup_datetime = datetime.combine(pickup_date, pickup_time)

        # Prepare features
        features = {
            "pickup_datetime": pickup_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "pickup_longitude": float(pickup_longitude),
            "pickup_latitude": float(pickup_latitude),
            "dropoff_longitude": float(dropoff_longitude),
            "dropoff_latitude": float(dropoff_latitude),
            "passenger_count": int(passenger_count),
            "vendor_id": int(vendor_id)
        }

        # Make prediction
        with st.spinner("Predicting..."):
            result = predict_manual(features)

        if result and 'predicted_duration' in result:
            prediction = result['predicted_duration']

            # Display prediction
            st.markdown("---")
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### Predicted Trip Duration")
            st.markdown(f'<div class="prediction-value">{prediction:.1f} seconds</div>', unsafe_allow_html=True)

            # Convert to minutes
            minutes = prediction / 60
            st.markdown(f"### ≈ {minutes:.1f} minutes", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Show input summary
            with st.expander("📋 Trip Summary"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Pickup:** ({pickup_latitude:.4f}, {pickup_longitude:.4f})")
                    st.write(f"**Datetime:** {pickup_datetime.strftime('%Y-%m-%d %H:%M')}")
                with col2:
                    st.write(f"**Dropoff:** ({dropoff_latitude:.4f}, {dropoff_longitude:.4f})")
                    st.write(f"**Passengers:** {passenger_count}")

elif "Test Sample" in page:
    st.header("🔍 Test Sample Prediction")
    st.write("Select from real test samples to see predictions vs actual values")

    # Fetch test samples
    with st.spinner("Loading test samples..."):
        test_df = get_test_samples()

    if test_df is not None and not test_df.empty:
        # Sample selection
        st.subheader("Select a Test Sample")

        # Display filters
        col1, col2, col3 = st.columns(3)

        with col1:
            passenger_filter = st.multiselect(
                "Passenger Count",
                options=sorted(test_df['passenger_count'].unique()),
                default=None
            )

        with col2:
            vendor_filter = st.multiselect(
                "Vendor ID",
                options=sorted(test_df['vendor_id'].unique()),
                default=None
            )

        # Apply filters
        filtered_df = test_df.copy()
        if passenger_filter:
            filtered_df = filtered_df[filtered_df['passenger_count'].isin(passenger_filter)]
        if vendor_filter:
            filtered_df = filtered_df[filtered_df['vendor_id'].isin(vendor_filter)]

        st.write(f"Showing {len(filtered_df)} samples")

        # Display sample selector
        if not filtered_df.empty:
            # Show first few samples in a table
            display_df = filtered_df.head(20).copy()
            display_df['pickup_datetime'] = pd.to_datetime(display_df['pickup_datetime']).dt.strftime('%Y-%m-%d %H:%M')

            st.dataframe(
                display_df[['sample_id', 'pickup_datetime', 'passenger_count',
                           'pickup_latitude', 'pickup_longitude',
                           'dropoff_latitude', 'dropoff_longitude', 'trip_duration']],
                hide_index=True,
                use_container_width=True
            )

            # Sample ID input
            sample_id = st.number_input(
                "Enter Sample ID from table above:",
                min_value=int(filtered_df['sample_id'].min()),
                max_value=int(filtered_df['sample_id'].max()),
                value=int(filtered_df['sample_id'].iloc[0])
            )

            if st.button("🔮 Predict Selected Sample", type="primary"):
                with st.spinner("Making prediction..."):
                    result = predict_test_sample(sample_id)

                if result:
                    # Display results
                    st.markdown("---")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Predicted Duration", f"{result['predicted']:.1f}s")
                        st.markdown('</div>', unsafe_allow_html=True)

                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Actual Duration", f"{result['actual']:.1f}s")
                        st.markdown('</div>', unsafe_allow_html=True)

                    with col3:
                        error = abs(result['predicted'] - result['actual'])
                        error_pct = (error / result['actual']) * 100
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Error", f"{error:.1f}s ({error_pct:.1f}%)")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Show details
                    with st.expander("📋 Sample Details"):
                        sample_data = filtered_df[filtered_df['sample_id'] == sample_id].iloc[0]

                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Pickup:**")
                            st.write(f"- Location: ({sample_data['pickup_latitude']:.4f}, {sample_data['pickup_longitude']:.4f})")
                            st.write(f"- Datetime: {sample_data['pickup_datetime']}")

                        with col2:
                            st.write("**Trip Info:**")
                            st.write(f"- Dropoff: ({sample_data['dropoff_latitude']:.4f}, {sample_data['dropoff_longitude']:.4f})")
                            st.write(f"- Passengers: {sample_data['passenger_count']}")
                            st.write(f"- Vendor: {sample_data['vendor_id']}")
        else:
            st.warning("No samples match the selected filters")
    else:
        st.error("Unable to load test samples. Please check if the API is running.")

else:  # Model Dashboard
    st.header("📈 Model Dashboard")

    # Get experiments
    experiments = get_experiments()

    if experiments:
        st.subheader("🧪 MLflow Experiments")

        for exp in experiments:
            with st.expander(f"📊 {exp.get('name', 'Unknown')}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Experiment ID:** {exp.get('experiment_id', 'N/A')}")
                    st.write(f"**Lifecycle Stage:** {exp.get('lifecycle_stage', 'N/A')}")

                with col2:
                    st.write(f"**Artifact Location:** {exp.get('artifact_location', 'N/A')}")
    else:
        st.info("No experiments found or unable to connect to MLflow")

    # Model info
    if model_info:
        st.markdown("---")
        st.subheader("🏆 Champion Model")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Model Version", model_info.get('model_version', 'N/A'))
            st.markdown('</div>', unsafe_allow_html=True)

        if 'metrics' in model_info:
            metrics = model_info['metrics']

            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("R² Score", f"{metrics.get('r2', 0):.4f}")
                st.markdown('</div>', unsafe_allow_html=True)

        # Additional model info
        with st.expander("📋 Model Details"):
            st.json(model_info)

    # System health
    st.markdown("---")
    st.subheader("💚 System Health")

    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()

            col1, col2, col3 = st.columns(3)

            with col1:
                mlflow_status = health_data.get('mlflow', False)
                if mlflow_status:
                    st.success("✅ MLflow Connected")
                else:
                    st.error("❌ MLflow Disconnected")

            with col2:
                airflow_status = health_data.get('airflow', False)
                if airflow_status:
                    st.success("✅ Airflow Connected")
                else:
                    st.error("❌ Airflow Disconnected")

            with col3:
                data_status = health_data.get('data_available', False)
                if data_status:
                    st.success("✅ Data Available")
                else:
                    st.error("❌ Data Missing")
        else:
            st.error("Unable to fetch health status")
    except:
        st.error("Health check failed")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>NYC Taxi Trip Duration Predictor | Powered by MLflow & FastAPI</p>
    </div>
    """,
    unsafe_allow_html=True
)
