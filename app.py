import streamlit as st
import pandas as pd
import numpy as np
import joblib
from prophet import Prophet
import lightgbm as lgb
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import os # Import os for path handling
import itertools # Import itertools for grid point generation

# Set page configuration
st.set_page_config(layout="wide", page_title="Chicago Crime Forecasting")

st.title("Chicago Crime Forecasting and Hotspot Analysis")
st.write("This application forecasts daily crime counts and identifies potential crime hotspots in Chicago.")

# --- Load Models and Data ---
# Define paths to saved models and data - adjust these paths as necessary
# Assuming models and data are saved in the current directory or a specified 'models' directory
MODELS_DIR = '.' # Or 'models' if you save them there

# Check if files exist before attempting to load
prophet_model_path = os.path.join(MODELS_DIR, 'prophet_model_refined.joblib')
lgbm_model_path = os.path.join(MODELS_DIR, 'lgbm_model_refined.joblib')
meta_model_path = os.path.join(MODELS_DIR, 'meta_model_refined.joblib')
rf_spatial_model_path = os.path.join(MODELS_DIR, 'rf_spatial_model.joblib')
grid_data_path = os.path.join(MODELS_DIR, 'grid_with_coords.geojson') # Assuming you saved the grid GeoDataFrame
# Assuming you saved the original df or a representative sample for spatial bin ranges
original_df_path = os.path.join(MODELS_DIR, 'df_cleaned.csv') # Path to cleaned original data or sample

prophet_model_refined = None
lgb_model_refined = None
meta_model_refined = None
rf_spatial_model = None
forecast_with_coords = None
original_df = None # To store original df or sample


@st.cache_resource # Cache the model loading
def load_models():
    loaded_models = {}
    try:
        # Prophet model requires special handling as it's not directly joblib serializable with regressors
        # A common workaround is to save/load using Prophet's internal methods or a custom pickle.
        # For demonstration, let's assume a simple case or a custom save/load for now.
        # In a real scenario, you might need to re-initialize and load parameters.
        # For this example, we'll assume a joblib compatible Prophet model or skip if not available.
        if os.path.exists(prophet_model_path):
             # If you saved the Prophet model in a joblib compatible way (e.g., using dill or a wrapper)
             try:
                 loaded_models['prophet'] = joblib.load(prophet_model_path)
                 st.sidebar.success("Prophet model loaded.")
             except Exception as e:
                 st.sidebar.warning(f"Could not load Prophet model: {e}")
                 loaded_models['prophet'] = None
        else:
             st.sidebar.warning("Prophet model file not found.")
             loaded_models['prophet'] = None

        if os.path.exists(lgbm_model_path):
            loaded_models['lgbm'] = joblib.load(lgbm_model_path)
            st.sidebar.success("LightGBM model loaded.")
        else:
             st.sidebar.warning("LightGBM model file not found.")
             loaded_models['lgbm'] = None

        if os.path.exists(meta_model_path):
            loaded_models['meta'] = joblib.load(meta_model_path)
            st.sidebar.success("Stacked Ensemble meta-model loaded.")
        else:
             st.sidebar.warning("Stacked Ensemble meta-model file not found.")
             loaded_models['meta'] = None

        if os.path.exists(rf_spatial_model_path):
            loaded_models['rf_spatial'] = joblib.load(rf_spatial_model_path)
            st.sidebar.success("Random Forest Spatial model loaded.")
        else:
             st.sidebar.warning("Random Forest Spatial model file not found.")
             loaded_models['rf_spatial'] = None


    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

    return loaded_models

@st.cache_data # Cache the data loading
def load_data():
    loaded_data = {}
    try:
        if os.path.exists(grid_data_path):
            # Assuming the grid data with estimated crime is saved as GeoJSON
            loaded_data['forecast_with_coords'] = gpd.read_file(grid_data_path)
            st.sidebar.success("Spatial grid data loaded.")
        else:
            st.sidebar.warning("Spatial grid data file not found.")
            loaded_data['forecast_with_coords'] = None

        if os.path.exists(original_df_path):
             loaded_data['original_df'] = pd.read_csv(original_df_path, parse_dates=['date'])
             st.sidebar.success("Original data (or sample) loaded for spatial bin ranges.")
        else:
             st.sidebar.warning("Original data file not found. Spatial prediction for specific time might be limited.")
             loaded_data['original_df'] = None


    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

    return loaded_data


loaded_models = load_models()
loaded_data = load_data()

prophet_model_refined = loaded_models.get('prophet')
lgb_model_refined = loaded_models.get('lgbm')
meta_model_refined = loaded_models.get('meta')
rf_spatial_model = loaded_models.get('rf_spatial')
forecast_with_coords = loaded_data.get('forecast_with_coords')
original_df = loaded_data.get('original_df')


# --- Time Series Forecasting Section ---
st.header("Daily Crime Count Forecasting")

if prophet_model_refined is not None and lgb_model_refined is not None and meta_model_refined is not None:
    st.write("Forecast the total number of crimes per day for a specified date range.")

    # Input widgets for date range
    start_date = st.date_input("Start Date", pd.to_datetime('today').date())
    end_date = st.date_input("End Date", pd.to_datetime('today').date() + pd.Timedelta(days=30))

    if start_date > end_date:
        st.error("Error: End date must fall after start date.")
    else:
        # Generate future dates for forecasting
        future_dates_ts = pd.DataFrame({'ds': pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='D'))})

        # Need historical data to compute lags/rolling means for the start of the forecast period.
        # Assuming daily_train_feat and daily_test_feat were saved or can be reconstructed.
        # For a real app, you'd need a robust way to get the last N days of actual data.
        # Let's assume we have access to a combined historical daily data up to the last date in training/testing.
        # Placeholder: In a real scenario, load or generate the necessary historical data.
        # For this example, we will assume we have 'daily_train_feat' and 'daily_test_feat' available from the notebook run.
        # If not available, this part will fail or produce dummy data.

        # Attempt to load historical daily data if not already in memory (e.g., from notebook run)
        # This is a fallback and might need adjustment based on how your notebook was run.
        historical_daily_path = os.path.join(MODELS_DIR, 'daily_crime_counts_with_features.csv') # Assuming you saved this
        historical_daily_data = None
        if os.path.exists(historical_daily_path):
             try:
                 historical_daily_data = pd.read_csv(historical_daily_path, parse_dates=['ds'])
                 st.sidebar.success("Historical daily data loaded for time series forecasting.")
             except Exception as e:
                 st.sidebar.warning(f"Could not load historical daily data: {e}")
                 historical_daily_data = None
        else:
             st.sidebar.warning("Historical daily data file not found. Time series feature engineering might be limited.")
             # If historical data is not available, we cannot compute true lags/rolling means.
             # This is a critical limitation for a production app without a data pipeline.
             # For this demo, we'll proceed with dummy feature generation if historical data is missing.


        def generate_ts_features(df_to_feat, historical_df=None):
            """Generates time series features (lags, rolling means, date features)."""
            df = df_to_feat.copy()

            if historical_df is not None:
                # Combine with historical data to compute features accurately
                combined_df = pd.concat([historical_df[['ds', 'y']], df[['ds', 'y']]], ignore_index=True)
                combined_df = combined_df.sort_values('ds').reset_index(drop=True)

                combined_df['lag_1'] = combined_df['y'].shift(1)
                combined_df['lag_3'] = combined_df['y'].shift(3)
                combined_df['lag_7'] = combined_df['y'].shift(7)
                combined_df['rolling_mean_7'] = combined_df['y'].rolling(window=7).mean()
                combined_df['rolling_std_7'] = combined_df['y'].rolling(window=7).std()

                # Extract features only for the dates in the original df_to_feat
                df = combined_df[combined_df['ds'].isin(df['ds'])].copy()
                df = df.set_index('ds').reindex(df_to_feat['ds']).reset_index() # Ensure original date order

            else:
                 # Fallback: Dummy or limited feature generation if no historical data
                 st.warning("Using dummy or limited features for time series forecast due to missing historical data.")
                 df['lag_1'] = df['y'].shift(1).fillna(method='bfill')
                 df['lag_3'] = df['y'].shift(3).fillna(method='bfill')
                 df['lag_7'] = df['y'].shift(7).fillna(method='bfill')
                 df['rolling_mean_7'] = df['y'].rolling(window=7).mean().fillna(method='bfill')
                 df['rolling_std_7'] = df['y'].rolling(window=7).std().fillna(method='bfill')


            df['day_of_week'] = df['ds'].dt.dayofweek
            df['week_of_year'] = df['ds'].dt.isocalendar().week.astype(int)
            df['month'] = df['ds'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

            return df

        # Generate features for the future dates using available historical data
        future_dates_feat = generate_ts_features(future_dates_ts, historical_daily_data)

        # Drop rows where essential features couldn't be computed (e.g., very start of historical data)
        future_dates_feat.dropna(subset=['lag_1', 'lag_3', 'lag_7', 'rolling_mean_7', 'rolling_std_7'], inplace=True)

        if not future_dates_feat.empty:
            # --- Make Predictions ---
            # Prophet requires a specific future dataframe format
            prophet_future = future_dates_feat[['ds', 'lag_1', 'lag_3', 'lag_7', 'rolling_mean_7', 'rolling_std_7']].copy()
            prophet_forecast = prophet_model_refined.predict(prophet_future)
            future_dates_feat['prophet_pred'] = prophet_forecast['yhat'].values

            # LightGBM requires the feature dataframe
            lgb_features_refined = ['lag_1', 'lag_3', 'lag_7', 'rolling_mean_7', 'rolling_std_7', 'day_of_week', 'week_of_year'] # Ensure features match training
            lgbm_pred = lgb_model_refined.predict(future_dates_feat[lgb_features_refined])
            future_dates_feat['lgbm_pred'] = np.maximum(0, lgbm_pred).round().astype(int) # Ensure non-negative integers

            # Stacked Ensemble requires predictions from base models + other features
            stack_X_future = pd.DataFrame({
                'prophet_pred': future_dates_feat['prophet_pred'].values,
                'lgbm_pred': future_dates_feat['lgbm_pred'].values,
                # Add other features used by the meta-model if any (e.g., rolling_std_7, lag_3, rolling_mean_7)
                'rolling_std_7': future_dates_feat['rolling_std_7'].values,
                'lag_3': future_dates_feat['lag_3'].values,
                'rolling_mean_7': future_dates_feat['rolling_mean_7'].values,
            })
            stack_pred = meta_model_refined.predict(stack_X_future)
            future_dates_feat['ensemble_pred'] = np.maximum(0, stack_pred).round().astype(int) # Ensure non-negative integers

            st.subheader("Forecasted Daily Crime Counts")
            st.line_chart(future_dates_feat.set_index('ds')[['prophet_pred', 'lgbm_pred', 'ensemble_pred']])

            st.subheader("Forecast Details")
            st.dataframe(future_dates_feat[['ds', 'prophet_pred', 'lgbm_pred', 'ensemble_pred']].rename(columns={
                'ds': 'Date',
                'prophet_pred': 'Prophet Forecast',
                'lgbm_pred': 'LightGBM Forecast',
                'ensemble_pred': 'Stacked Ensemble Forecast'
            }).set_index('Date'))

        else:
             st.warning("Could not generate sufficient features for the selected date range. Please check historical data availability.")


else:
     st.warning("Time series forecasting models not loaded. Please ensure model files are in the correct directory.")


# --- Spatial Hotspot Analysis Section ---
st.header("Crime Hotspot Analysis")

if rf_spatial_model is not None and forecast_with_coords is not None:
    st.write("Identify potential crime hotspots based on location and time of day.")

    # Input widgets for spatial analysis (e.g., specific hour, day of week)
    # Or, use the pre-computed averaged heatmap if that's what forecast_with_coords represents
    analysis_type = st.radio("Select analysis type:", ("Averaged Hotspots (Historical Proportion)", "Predict for Specific Time"))

    if analysis_type == "Averaged Hotspots (Historical Proportion)":
        st.write("Displaying crime probability hotspots averaged over historical time features.")
        # Use the pre-computed forecast_with_coords data

        if not forecast_with_coords.empty:
            # Create a Folium map
            # Use the mean of the grid centroids for map center
            map_center_lat = forecast_with_coords['lat'].mean() if 'lat' in forecast_with_coords.columns and not forecast_with_coords['lat'].isnull().all() else 41.8781
            map_center_lon = forecast_with_coords['lon'].mean() if 'lon' in forecast_with_coords.columns and not forecast_with_coords['lon'].isnull().all() else -87.6298

            m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=11)

            # Add HeatMap layer
            heat_data = [[row['lat'], row['lon'], row['crime_probability']]
                         for _, row in forecast_with_coords[forecast_with_coords['crime_probability'] > 0].iterrows()]

            if heat_data:
                 HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
                 st.subheader("Average Crime Probability Heatmap")
                 # Display the map
                 st.components.v1.html(m._repr_html_(), width=700, height=500)
            else:
                 st.info("No spatial crime probability data to display heatmap.")

            # Optionally display top hotspots
            top_k_hotspots = st.slider("Show Top K Hotspots", 0, len(forecast_with_coords), 50)
            if top_k_hotspots > 0:
                 top_hotspots = forecast_with_coords.sort_values(by='crime_probability', ascending=False).head(top_k_hotspots)
                 st.subheader(f"Top {top_k_hotspots} Hotspot Grid Cells (Averaged)")
                 st.dataframe(top_hotspots[['lat', 'lon', 'crime_probability']].rename(columns={
                     'lat': 'Latitude', 'lon': 'Longitude', 'crime_probability': 'Average Crime Probability'
                 })) # Exclude geometry column from display


        else:
            st.warning("Spatial grid data is empty or not loaded correctly.")


    elif analysis_type == "Predict for Specific Time":
        st.write("Predict crime probability for spatial bins at a specific hour and day of the week.")

        # Input widgets for specific time
        selected_hour = st.slider("Select Hour of Day (0-23)", 0, 23, 12)
        selected_day_of_week = st.selectbox("Select Day of Week", options=range(7), format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])

        if rf_spatial_model is not None and original_df is not None:
            # Create a grid of spatial bins for the selected time
            # Need lat_bin and lon_bin values from the original data or the grid
            if not original_df.empty and 'lat_bin' in original_df.columns and 'lon_bin' in original_df.columns:
                lat_bins_pred = np.round(np.linspace(original_df['lat_bin'].min(), original_df['lat_bin'].max(), 50), 3)
                lon_bins_pred = np.round(np.linspace(original_df['lon_bin'].min(), original_df['lon_bin'].max(), 50), 3)

                grid_points_pred = list(itertools.product(lat_bins_pred, lon_bins_pred))
                grid_df_pred = pd.DataFrame(grid_points_pred, columns=['lat_bin', 'lon_bin'])

                # Add the selected time features
                grid_df_pred['hour'] = selected_hour
                grid_df_pred['day_of_week'] = selected_day_of_week

                # Define features used by the spatial RF model
                spatial_features = ['lat_bin', 'lon_bin', 'hour', 'day_of_week']

                # Predict probabilities using the spatial RF model
                probs_pred = rf_spatial_model.predict_proba(grid_df_pred[spatial_features])[:, 1]
                grid_df_pred['crime_prob'] = probs_pred

                # Convert to GeoDataFrame for mapping
                geometry_pred = [Point(xy) for xy in zip(grid_df_pred['lon_bin'], grid_df_pred['lat_bin'])]
                gdf_pred = gpd.GeoDataFrame(grid_df_pred, geometry=geometry_pred, crs="EPSG:4326")


                # Plot heatmap for the specific time
                if not gdf_pred.empty:
                    # Calculate map center from the predicted grid points
                    map_center_lat_pred = gdf_pred['lat_bin'].mean() if 'lat_bin' in gdf_pred.columns and not gdf_pred['lat_bin'].isnull().all() else 41.8781
                    map_center_lon_pred = gdf_pred['lon_bin'].mean() if 'lon_bin' in gdf_pred.columns and not gdf_pred['lon_bin'].isnull().all() else -87.6298

                    m_pred = folium.Map(location=[map_center_lat_pred, map_center_lon_pred], zoom_start=11)

                    heat_data_pred = [[row['lat_bin'], row['lon_bin'], row['crime_prob']]
                                     for _, row in gdf_pred[gdf_pred['crime_prob'] > 0].iterrows()]

                    if heat_data_pred:
                         HeatMap(heat_data_pred, radius=15, blur=10, max_zoom=1).add_to(m_pred)
                         st.subheader(f"Crime Probability Heatmap for {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][selected_day_of_week]} at Hour {selected_hour}")
                         st.components.v1.html(m_pred._repr_html_(), width=700, height=500)
                    else:
                         st.info("No spatial crime probability data to display heatmap for this time.")

                    # Optionally display top hotspots for the specific time
                    top_k_hotspots_time = st.slider("Show Top K Hotspots for this time", 0, len(gdf_pred), 50)
                    if top_k_hotspots_time > 0:
                         top_hotspots_time = gdf_pred.sort_values(by='crime_prob', ascending=False).head(top_k_hotspots_time)
                         st.subheader(f"Top {top_k_hotspots_time} Hotspot Grid Cells for this time")
                         st.dataframe(top_hotspots_time[['lat_bin', 'lon_bin', 'crime_prob']].rename(columns={
                             'lat_bin': 'Latitude Bin', 'lon_bin': 'Longitude Bin', 'crime_prob': 'Crime Probability'
                         }))

                else:
                     st.warning("Could not generate spatial grid for prediction with selected time.")

            else:
                st.warning("Original dataframe 'df' with 'lat_bin' and 'lon_bin' not found or is empty. Cannot generate spatial grid for prediction.")


        else:
             st.warning("Random Forest Spatial model or original data not loaded. Cannot perform specific time prediction.")

else:
     st.warning("Spatial analysis models or data not loaded. Please ensure model and grid data files are in the correct directory.")


st.sidebar.header("About")
st.sidebar.info(
    "This app demonstrates forecasting and hotspot analysis of Chicago crime data "
    "using time series (Prophet, LightGBM, Stacked Ensemble) and spatial (Random Forest) models."
)
