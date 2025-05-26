import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import uuid
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
import json
import geopandas as gpd
from folium.plugins import MarkerCluster
import plotly.subplots as sp
from PIL import Image
import requests
import pytz as dtlib

# Hide the default Streamlit navigation menu
st.set_page_config(
    page_title="Energy Disaggregation Model Output",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Add Font Awesome CSS
st.markdown('''<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">''', unsafe_allow_html=True)

# Hide Streamlit's default menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Set custom theme colors
# Primary Colors
primary_purple = "#515D9A"
light_purple = "#B8BCF3"
white = "#FFFFFF"

# Secondary Background Colors
dark_purple = "#202842"
dark_gray = "#222222"

# Secondary Highlight Colors
cream = "#F0E0C3"
salmon = "#E99371"
green = "#6EA199"
red = "#D3745F"

# Apply custom CSS for theming 
st.markdown(f"""
<style>
    /* Main background */
    .stApp {{
        background-color: {white};
        color: {dark_purple};
    }}
    
    /* Sidebar */
    .css-1d391kg {{
        background-color: {primary_purple};
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {primary_purple} !important;
    }}
    
    /* Text */
    p {{
        color: {dark_purple};
    }}
    
    /* Caption */
    .caption {{
        color: {primary_purple};
    }}
    
    /* Expander headers */
    .streamlit-expanderHeader {{
        color: {white} !important;
        background-color: {primary_purple};
    }}
    
    /* Dataframe */
    .dataframe-container {{
        color: {dark_gray};
    }}
    
    /* Metric labels - even larger font size */
    [data-testid="stMetricLabel"] {{
        color: {dark_purple} !important;
        font-weight: bold;
        font-size: 1.8rem !important;
    }}
    
    /* Metric values - even larger font size */
    [data-testid="stMetricValue"] {{
        color: {dark_purple} !important;
        font-size: 2.0rem !important;
        font-weight: bold;
    }}
    
    /* Metric delta - make it larger too */
    [data-testid="stMetricDelta"] {{
        color: {dark_purple} !important;
        font-size: 1.5rem !important;
    }}
    
    /* Selectbox */
    .stSelectbox {{
        color: {dark_gray};
    }}
    
    /* Caption text */
    .css-1t42vg8 {{
        color: {primary_purple};
    }}
    
    /* Banner styling */
    .banner-container {{
        width: 100%;
        margin-bottom: 1rem;
    }}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
    /* Hide Streamlit Navigation Elements */
    .stSidebarNav {
        display: none !important;
    }
    div[data-testid="stSidebarNav"] {
        display: none !important;
    }
    div[data-testid="stSidebarNavItems"] {
        display: none !important;
    }
    .st-emotion-cache-79elbk {
        display: none !important;
    }
    .st-emotion-cache-1rtdyuf {
        display: none !important;
    }
    .st-emotion-cache-6tkfeg {
        display: none !important;
    }
    .st-emotion-cache-eczjsme {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Force theme to stay consistent regardless of system settings
st.markdown("""
<style>
    /* Force light theme and override dark mode detection */
    .stApp {
        background-color: #FFFFFF !important;
    }
    
    /* Override sidebar in both modes */
    [data-testid="stSidebar"] {
        background-color: #B8BCF3 !important;
        color: #FFFFFF !important;
    }
    
    /* Sidebar elements */
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stExpander,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #FFFFFF !important;
    }
    
    /* Sidebar expander */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        color: #FFFFFF !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Sidebar selectbox/multiselect dropdown */
    .stSidebar .stMultiSelect div[data-baseweb="select"] span {
        color: #202842 !important;
    }
    
    /* Force dark purple text everywhere in main content */
    .main p, .main li, .main label {
        color: #202842 !important;
    }
    
    /* Override any automatic color scheme changes */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #FFFFFF !important;
        }
        
        .main p, .main li, .main label, .main div {
            color: #202842 !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #515D9A !important;
        }
    }
    
    /* Radio buttons in sidebar */
    [data-testid="stSidebar"] .stRadio label span {
        color: #FFFFFF !important;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: rgba(81, 93, 154, 0.2) !important;
        color: #202842 !important;
    }
    
    /* Table styling */
    .dataframe th {
        background-color: #515D9A !important;
        color: white !important;
    }
    
    .dataframe td {
        background-color: #FFFFFF !important;
        color: #202842 !important;
    }
</style>
""", unsafe_allow_html=True)

# Define the banner path once at the top of the script
banner_path = "ECMX_linkedinheader_SN.png"  

# Helper function for generating random IDs
def generate_random_id(prefix="ID_"):
    return f"{prefix}{str(uuid.uuid4().hex)[:8].upper()}"

# Helper function for formatting timestamps 
def format_timestamp_column(df, column_name, target_format='%Y-%m-%d', is_unix=True):
    if column_name not in df.columns:
        # st.warning(f"Timestamp column '{column_name}' not found in DataFrame for formatting.")
        return df # Return df unmodified if column not found
    
    temp_col = df[column_name].copy()
    formatted_col_name = column_name + '_formatted'
    
    if is_unix:
        temp_col = pd.to_numeric(temp_col, errors='coerce')
        dt_series = pd.to_datetime(temp_col, unit='s', errors='coerce')
    else:
        dt_series = pd.to_datetime(temp_col, errors='coerce')
        
    df[formatted_col_name] = dt_series.dt.strftime(target_format)
    # Handle NaT explicitly after attempting to format
    # NaNs in dt_series (from coercion errors) will lead to NaNs here, fill them.
    df[formatted_col_name] = df[formatted_col_name].fillna('Invalid Date')
    
    # Check if all valid datetimes became 'Invalid Date' due to strftime issue (unlikely with common formats)
    # or if the source was largely unparseable.
    if not dt_series.isna().all() and df[formatted_col_name].replace('Invalid Date', pd.NaT).isna().all():
         st.warning(f"All valid entries in '{column_name}' resulted in 'Invalid Date' after formatting. Check target_format or source data.")
    elif dt_series.isna().sum() > 0 and (df[formatted_col_name] == 'Invalid Date').sum() > dt_series.isna().sum() :
         # This means some non-NaT datetimes also failed to format.
         pass # Potentially log this if it's a concern.

    return df

# Add page selection at the top (now with three options)
page = st.sidebar.radio("Select Page", ["Sample Output", "Performance Metrics", "Interactive Map"], index=0) # Default to Sample Output

# Display banner on all pages
try:
    # Display banner image
    banner_image = Image.open(banner_path)
    st.image(banner_image, use_container_width=True)
except Exception as e:
    st.warning(f"Banner image not found at {banner_path}. Please update the path in the code.")

# Initialize selected_model to the default value before page logic
selected_model = "V6" # Default model

if page == "Sample Output":
    st.title("Energy Disaggregation Model: Sample Output")
    st.markdown("""
    This page displays sample outputs from the energy disaggregation model. 
    It includes detailed event windows and a monthly breakdown of appliance consumption.
    The data presented is based on a small cohort of customers (9-10) for illustrative purposes.
    """)

    # Add disclaimer about data anonymization
    st.markdown("""
    <i class="fas fa-user-secret"></i> **Data Anonymization Note:** 
    To protect privacy, `custid` and `meterid` values in the tables below have been replaced with randomly generated identifiers. 
    The underlying data relationships are maintained.
    """, unsafe_allow_html=True)

    # --- Load and Anonymize Data ---
    @st.cache_data
    def load_and_anonymize_data(details_path, meters_path, cust_id_col_name='custid', meter_id_col_name='meter_id'):
        try:
            df_details = pd.read_csv(details_path)
            df_meters = pd.read_csv(meters_path)
        except FileNotFoundError as e:
            st.error(f"Error: One or both data files not found. Please check paths: {e}")
            return pd.DataFrame(), pd.DataFrame(), {}
        except Exception as e:
            st.error(f"Error loading CSV files: {e}")
            return pd.DataFrame(), pd.DataFrame(), {}

        # Ensure custid and meter_id are strings for consistent processing
        for df, name in [(df_details, "Details"), (df_meters, "Meters")]:
            for col, expected_col_name in [(cust_id_col_name, 'Customer ID'), (meter_id_col_name, 'Meter ID')]:
                if col in df.columns:
                    df[col] = df[col].astype(str)
                else:
                    st.warning(f"Column '{col}' (expected for {expected_col_name}) not found in {name} data.")

        all_custids = pd.Series(dtype=str)
        if cust_id_col_name in df_details.columns:
            all_custids = pd.concat([all_custids, df_details[cust_id_col_name]], ignore_index=True)
        if cust_id_col_name in df_meters.columns:
            all_custids = pd.concat([all_custids, df_meters[cust_id_col_name]], ignore_index=True)
        
        unique_custids = all_custids.dropna().unique()
        custid_map = {orig_id: generate_random_id("CUST_") for orig_id in unique_custids}

        all_meterids = pd.Series(dtype=str)
        if meter_id_col_name in df_details.columns:
            all_meterids = pd.concat([all_meterids, df_details[meter_id_col_name]], ignore_index=True)
        if meter_id_col_name in df_meters.columns:
             all_meterids = pd.concat([all_meterids, df_meters[meter_id_col_name]], ignore_index=True)

        unique_meterids = all_meterids.dropna().unique()
        # Store map against the original column name for clarity if needed later, though generation is universal
        meterid_map = {orig_id: generate_random_id("MTR_") for orig_id in unique_meterids}
        
        id_maps = {cust_id_col_name: custid_map, meter_id_col_name: meterid_map}

        # Create anonymized columns
        if cust_id_col_name in df_details.columns:
            df_details['custid_anon'] = df_details[cust_id_col_name].map(custid_map)
        if meter_id_col_name in df_details.columns:
            df_details['meterid_anon'] = df_details[meter_id_col_name].map(meterid_map)
        
        if cust_id_col_name in df_meters.columns:
            df_meters['custid_anon'] = df_meters[cust_id_col_name].map(custid_map)
        if meter_id_col_name in df_meters.columns:
            df_meters['meterid_anon'] = df_meters[meter_id_col_name].map(meterid_map)
            
        return df_details, df_meters, id_maps

    details_file = "disagg_details_new_cohort_updated_1.csv"
    meters_file = "disagg_meters_new_cohort_updated_1.csv"
    # Pass the correct column names for meter_id
    df_details_orig, df_meters_orig, id_maps = load_and_anonymize_data(
        details_file, meters_file, cust_id_col_name='custid', meter_id_col_name='meter_id'
    )

    if df_details_orig.empty and df_meters_orig.empty: # Only stop if both are empty due to load error
        st.warning("Could not load data files. Ensure they are in the correct location and format: " + f"'{details_file}', '{meters_file}'")
        st.stop()
    elif df_details_orig.empty:
        st.warning(f"Details data file '{details_file}' is empty or could not be loaded properly.")
    elif df_meters_orig.empty:
        st.warning(f"Meters data file '{meters_file}' is empty or could not be loaded properly.")

    # --- REORDERED: Table 2 (Monthly Appliance Breakdown) first ---
    st.subheader("Table 2: Monthly Appliance Energy Breakdown")
    st.markdown("This table summarizes the estimated monthly energy consumption or generation for each appliance, compared to the total meter consumption for that month.")

    # Initialize df_table2_final to an empty DataFrame to ensure it exists
    df_table2_final = pd.DataFrame()

    if not df_details_orig.empty: # Table 2 now primarily relies on details_orig
        df_details_for_table2 = df_details_orig.copy()
        # df_meters_for_table2 = df_meters_orig.copy() # Still loaded if needed for other things, but not primary for table 2 grid

        required_details_cols_t2 = [
            'custid_anon', 'meterid_anon', 'equipment_disagg_start', 
            'equipment_type', 'equipment_consumption', 'direction', 'meter_consumption'
        ]
        missing_req_details_cols = [col for col in required_details_cols_t2 if col not in df_details_for_table2.columns]

        if not missing_req_details_cols:
            temp_ws_col = pd.to_numeric(df_details_for_table2['equipment_disagg_start'], errors='coerce')
            df_details_for_table2['reference_month_dt'] = pd.to_datetime(temp_ws_col, unit='s', errors='coerce')
            df_details_for_table2['reference_month_str'] = df_details_for_table2['reference_month_dt'].dt.strftime('%Y-%m')

            df_details_for_table2['equipment_consumption'] = pd.to_numeric(df_details_for_table2['equipment_consumption'], errors='coerce').fillna(0)
            df_details_for_table2['meter_consumption'] = pd.to_numeric(df_details_for_table2['meter_consumption'], errors='coerce').fillna(0)

            appliance_energy_long = df_details_for_table2.groupby(
                ['custid_anon', 'meterid_anon', 'reference_month_str', 'equipment_type', 'direction'],
                as_index=False
            )['equipment_consumption'].sum()
            appliance_energy_long = appliance_energy_long.rename(columns={'equipment_consumption': 'energy_wh_appliance'})
            
            monthly_total_meter_consumption = df_details_for_table2.groupby(
                ['custid_anon', 'meterid_anon', 'reference_month_str'],
                as_index=False
            )['meter_consumption'].sum().rename(columns={'meter_consumption': 'grid_total_wh'})

            if not appliance_energy_long.empty and not monthly_total_meter_consumption.empty:
                df_table2_merged = pd.merge(
                    appliance_energy_long,
                    monthly_total_meter_consumption,
                    on=['custid_anon', 'meterid_anon', 'reference_month_str'],
                    how='left'
                )
                df_table2_merged['grid_total_wh'] = df_table2_merged['grid_total_wh'].fillna(0)
                df_table2_merged['energy_wh_appliance'] = pd.to_numeric(df_table2_merged['energy_wh_appliance'], errors='coerce').fillna(0)

                df_table2_qc = df_table2_merged.copy()
                df_table2_qc['consumption_energy_val'] = df_table2_qc.apply(
                    lambda row: row['energy_wh_appliance'] if str(row['direction']).lower() != 'generation' else 0, axis=1
                )

                def qc_consumption_group(group):
                    grid_value = group['grid_total_wh'].iloc[0]
                    total_identified_consumption = group['consumption_energy_val'].sum()
                    if grid_value > 0 and total_identified_consumption > grid_value:
                        if total_identified_consumption > 0:
                            scaling_factor = grid_value / total_identified_consumption
                            group.loc[group['direction'].str.lower() != 'generation', 'energy_wh_appliance'] = group.loc[group['direction'].str.lower() != 'generation', 'energy_wh_appliance'] * scaling_factor
                        else: pass
                    elif grid_value <= 0:
                        group.loc[group['direction'].str.lower() != 'generation', 'energy_wh_appliance'] = 0
                    return group

                grouped_for_qc = df_table2_qc.groupby(['custid_anon', 'meterid_anon', 'reference_month_str'], group_keys=False)
                df_table2_final = grouped_for_qc.apply(qc_consumption_group).reset_index(drop=True)
                
                df_table2_final['month_dt_display'] = pd.to_datetime(df_table2_final['reference_month_str'] + '-01', errors='coerce')
                df_table2_final['Month'] = df_table2_final['month_dt_display'].dt.strftime('%B %Y')
                df_table2_final['Month'] = df_table2_final['Month'].fillna('Invalid Month')

                cols_table2_display_map = {
                    'custid_anon': 'Customer ID (Anon)',
                    'meterid_anon': 'Meter ID (Anon)',
                    'Month': 'Month',
                    'equipment_type': 'Appliance',
                    'energy_wh_appliance': 'Energy (Wh) [QC]',
                    'direction': 'Direction',
                    'grid_total_wh': 'Total Meter (Wh)'
                }
                if 'equipment_type' not in df_table2_final.columns and 'appliance_type' in df_table2_final.columns:
                     cols_table2_display_map['appliance_type'] = cols_table2_display_map.pop('equipment_type')
                elif 'equipment_type' not in df_table2_final.columns:
                     st.warning("Could not find appliance type column for Table 2 display map.")

                actual_cols_table2 = [col for col in cols_table2_display_map.keys() if col in df_table2_final.columns]
                display_df_table2 = df_table2_final[actual_cols_table2].rename(columns=cols_table2_display_map)
                
                if actual_cols_table2:
                    if 'Appliance' not in display_df_table2.columns and 'equipment_type' in display_df_table2.columns:
                        display_df_table2 = display_df_table2.rename(columns={'equipment_type':'Appliance'})
                    elif 'Appliance' not in display_df_table2.columns and 'appliance_type' in display_df_table2.columns:
                        display_df_table2 = display_df_table2.rename(columns={'appliance_type':'Appliance'})
                    st.dataframe(display_df_table2, height=300, use_container_width=True) # Adjusted height for Table 2
                    missing_cols_t2 = [v for k,v in cols_table2_display_map.items() if k not in actual_cols_table2]
                    if missing_cols_t2:
                        st.caption(f"Note: The following expected columns are not shown as they were not found in the processed data: {', '.join(missing_cols_t2)}.")
                else:
                    st.info("No data to display for the monthly appliance breakdown after processing.")
            else:
                st.info("Could not generate Table 2 due to missing processed data from appliance energy or monthly total meter consumption.")
        else:
            st.error(f"Details data is missing one or more required columns for Table 2: {missing_req_details_cols}. Table 2 cannot be generated.")
    # Removed the elif for df_meters_orig.empty as Table 2 doesn't primarily depend on it now
    elif df_details_orig.empty:
        st.info("Table 2 (Monthly Breakdown) cannot be generated as the details data file could not be loaded or is empty.")
    else:
        st.error("An unexpected issue occurred preparing data for Table 2.")

    st.markdown("---") # Separator after Table 2

    # --- REORDERED: Table 1 (Disaggregation Event Windows) second ---
    st.subheader("Table 1: Disaggregation Event Windows")
    st.markdown("This table shows individual appliance usage events detected by the model.")
    
    if not df_details_orig.empty:
        df_details_table1 = df_details_orig.copy()
        df_details_table1 = format_timestamp_column(df_details_table1, 'equipment_disagg_start', '%Y-%m-%d %H:%M:%S', is_unix=True)
        df_details_table1 = format_timestamp_column(df_details_table1, 'equipment_disagg_end', '%Y-%m-%d %H:%M:%S', is_unix=True)
        cols_table1_display_map = {
            'custid_anon': 'Customer ID (Anon)',
            'meterid_anon': 'Meter ID (Anon)',
            'equipment_disagg_start_formatted': 'Event Start',
            'equipment_disagg_end_formatted': 'Event Stop',
            'equipment_type': 'Appliance',
            'equipment_detected': 'Status',
            'direction': 'Direction',
            'equipment_consumption': 'Energy (Wh)',
            'uom': 'UOM',
            'interval_count': 'Interval Count'
        }
        actual_cols_table1 = [col for col in cols_table1_display_map.keys() if col in df_details_table1.columns]
        if actual_cols_table1:
            st.dataframe(df_details_table1[actual_cols_table1].rename(columns=cols_table1_display_map), height=300, use_container_width=True) # Adjusted height for Table 1
            missing_cols_t1 = [v for k, v in cols_table1_display_map.items() if k not in actual_cols_table1]
            if missing_cols_t1:
                 st.caption(f"Note: The following expected columns are not shown as they were not found in the source data: {', '.join(missing_cols_t1)}.")
        else:
            st.warning("Not enough data columns found in the details file to display Table 1.")
    else:
        st.info(f"No data loaded from '{details_file}' to display Table 1.")

    st.markdown("---") # Separator after Table 1

    # --- Appliance Usage Heatmaps ---
    st.subheader("Appliance Usage Heatmaps")

    if not df_details_orig.empty and 'equipment_type' in df_details_orig.columns and 'equipment_disagg_start' in df_details_orig.columns and 'equipment_consumption' in df_details_orig.columns:
        df_heatmaps = df_details_orig.copy()
        df_heatmaps['equipment_consumption'] = pd.to_numeric(df_heatmaps['equipment_consumption'], errors='coerce').fillna(0)
        df_heatmaps['timestamp'] = pd.to_datetime(df_heatmaps['equipment_disagg_start'], unit='s', errors='coerce')
        df_heatmaps = df_heatmaps.dropna(subset=['timestamp']) # Drop rows where timestamp conversion failed

        available_appliances = sorted(df_heatmaps['equipment_type'].dropna().unique())
        if not available_appliances:
            st.info("No appliance data available to generate heatmaps.")
        else:
            selected_appliance_heatmap = st.selectbox(
                "Select Appliance for Heatmap:", 
                options=available_appliances,
                index=0 if available_appliances else -1 # Handle empty list
            )

            if selected_appliance_heatmap:
                df_appliance_specific = df_heatmaps[df_heatmaps['equipment_type'] == selected_appliance_heatmap].copy()

                if df_appliance_specific.empty:
                    st.info(f"No data found for appliance '{selected_appliance_heatmap}' to generate heatmaps.")
                else:
                    tab_daily, tab_monthly = st.tabs(["Daily View Heatmap", "Monthly View Heatmap"])

                    with tab_daily:
                        st.markdown(f"**Hourly Energy Consumption for {selected_appliance_heatmap} (Wh) - Daily View**")
                        df_appliance_specific['date'] = df_appliance_specific['timestamp'].dt.date
                        df_appliance_specific['hour'] = df_appliance_specific['timestamp'].dt.hour
                        
                        daily_heatmap_data = df_appliance_specific.groupby(['date', 'hour'])['equipment_consumption'].sum().unstack(fill_value=0)
                        
                        if not daily_heatmap_data.empty:
                            # Ensure all hours 0-23 are present, even if no consumption
                            all_hours = pd.Index(range(24), name='hour')
                            daily_heatmap_data = daily_heatmap_data.reindex(columns=all_hours, fill_value=0)
                            # Sort rows by date
                            daily_heatmap_data = daily_heatmap_data.sort_index()

                            fig_daily = px.imshow(
                                daily_heatmap_data,
                                labels=dict(x="Hour of Day", y="Date", color="Energy (Wh)"),
                                x=daily_heatmap_data.columns, # Hours
                                y=daily_heatmap_data.index,  # Dates
                                aspect="auto",
                                color_continuous_scale=px.colors.sequential.Viridis
                            )
                            fig_daily.update_xaxes(side="bottom", tickmode='array', tickvals=list(range(24)), ticktext=[str(h) for h in range(24)])
                            fig_daily.update_layout(title_text='', title_x=0.5, height=max(400, 50 + len(daily_heatmap_data.index) * 20))
                            st.plotly_chart(fig_daily, use_container_width=True)
                        else:
                            st.info(f"No daily consumption data to display for {selected_appliance_heatmap}.")

                    with tab_monthly:
                        st.markdown(f"**Daily Energy Consumption for {selected_appliance_heatmap} (Wh) - Monthly View**")
                        df_appliance_specific['year_month'] = df_appliance_specific['timestamp'].dt.to_period('M').astype(str)
                        df_appliance_specific['day_of_month'] = df_appliance_specific['timestamp'].dt.day
                        
                        monthly_heatmap_data = df_appliance_specific.groupby(['year_month', 'day_of_month'])['equipment_consumption'].sum().unstack(fill_value=0)

                        if not monthly_heatmap_data.empty:
                            # Ensure all days 1-31 are present
                            all_days = pd.Index(range(1, 32), name='day_of_month')
                            monthly_heatmap_data = monthly_heatmap_data.reindex(columns=all_days, fill_value=0)
                            # Sort rows by month
                            monthly_heatmap_data = monthly_heatmap_data.sort_index()

                            fig_monthly = px.imshow(
                                monthly_heatmap_data,
                                labels=dict(x="Day of Month", y="Month", color="Energy (Wh)"),
                                x=monthly_heatmap_data.columns, # Days
                                y=monthly_heatmap_data.index,  # Year-Months
                                aspect="auto",
                                color_continuous_scale=px.colors.sequential.Plasma
                            )
                            fig_monthly.update_xaxes(side="bottom", tickmode='array', tickvals=list(range(1,32)), ticktext=[str(d) for d in range(1,32)])
                            fig_monthly.update_layout(title_text='', title_x=0.5, height=max(400, 50 + len(monthly_heatmap_data.index) * 30))
                            st.plotly_chart(fig_monthly, use_container_width=True)
                        else:
                            st.info(f"No monthly consumption data to display for {selected_appliance_heatmap}.")
            else:
                st.info("Please select an appliance to view heatmaps.")
    else:
        st.info("Heatmap data requirements not met. Ensure 'equipment_type', 'equipment_disagg_start', and 'equipment_consumption' columns are present in the details CSV.")

    st.markdown("---")
    st.caption("Data presented is for illustrative purposes. Energy values are estimates from the model. 'Energy (Wh) [QC]' indicates values have been quality-controlled against total meter consumption for the month.")

elif page == "Performance Metrics":
    # Performance Metrics page
    # Just one title, with a more comprehensive subheader
    st.title("NILM Performance Dashboard")
    st.subheader(f"Device Detection Performance Analysis")
    
    # Define metrics data with updated values
    @st.cache_data
    def load_performance_data():
        # Data for all models
        models_data = {
            'V1': {
                'DPSPerc': {
                    'EV Charging': 80.2603,
                    'AC Usage': 76.6360,
                    'PV Usage': 92.5777
                },
                'FPR': {
                    'EV Charging': 0.1961,
                    'AC Usage': 0.3488,
                    'PV Usage': 0.1579
                },
                'TECA': {
                    'EV Charging': 0.7185,
                    'AC Usage': 0.6713,
                    'PV Usage': 0.8602
                }
            },
            'V2': {
                'DPSPerc': {
                    'EV Charging': 82.4146,
                    'AC Usage': 76.6360,
                    'PV Usage': 92.5777
                },
                'FPR': {
                    'EV Charging': 0.1667,
                    'AC Usage': 0.3488,
                    'PV Usage': 0.1579
                },
                'TECA': {
                    'EV Charging': 0.7544,
                    'AC Usage': 0.6519,
                    'PV Usage': 0.8812
                }
            },
            'V3': {
                'DPSPerc': {
                    'EV Charging': 81.8612,
                    'AC Usage': 73.0220,
                    'PV Usage': 92.0629
                },
                'FPR': {
                    'EV Charging': 0.1667,
                    'AC Usage': 0.4419,
                    'PV Usage': 0.1754
                },
                'TECA': {
                    'EV Charging': 0.7179,
                    'AC Usage': 0.6606,
                    'PV Usage': 0.8513
                }
            },
            'V4': {
                'DPSPerc': {
                    'EV Charging': 85.8123,
                    'AC Usage': 76.3889,
                    'PV Usage': 91.5448
                },
                'FPR': {
                    'EV Charging': 0.1176,
                    'AC Usage': 0.1977,
                    'PV Usage': 0.1930
                },
                'TECA': {
                    'EV Charging': 0.7741,
                    'AC Usage': 0.6718,
                    'PV Usage': 0.8395
                }
            },
            'V5': {
                'DPSPerc': {
                    'EV Charging': 81.1060,
                    'AC Usage': 77.5813,
                    'PV Usage': 91.9317
                },
                'FPR': {
                    'EV Charging': 0.1373,
                    'AC Usage': 0.3750,
                    'PV Usage': 0.1930
                },
                'TECA': {
                    'EV Charging': 0.6697,
                    'AC Usage': 0.6912,
                    'PV Usage': 0.7680
                }
            },
            'V6': {
                'DPSPerc': {
                    'EV Charging': 86.5757,
                    'AC Usage': 83.0483,
                    'PV Usage': 85.0008
                },
                'FPR': {
                    'EV Charging': 0.0541,
                    'AC Usage': 0.3750,
                    'PV Usage': 0.0000
                },
                'TECA': {
                    'EV Charging': 0.7491,
                    'AC Usage': 0.7739,
                    'PV Usage': 0.6443
                }
            }
        }
        
        return models_data

    # Load performance data
    models_data = load_performance_data()

    # Define model dates
    model_dates = {
        "V1": "March 3",
        "V2": "March 6",
        "V3": "March 7",
        "V4": "March 13",
        "V5": "March 18",
        "V6": "March 25"
    }

    # Model selection in sidebar
    selected_model = st.sidebar.selectbox(
        "Select Model",
        ["V1", "V2", "V3", "V4", "V5", "V6"],
        index=5,  # Default to V6
        format_func=lambda x: f"{x} ({model_dates[x]})"  # Format with dates
    )

    # Add device type filter in sidebar - remove WH Usage
    device_types = st.sidebar.multiselect(
        "Select Device Types",
        ["EV Charging", "AC Usage", "PV Usage"],
        default=["EV Charging", "AC Usage", "PV Usage"]
    )

    # Add fallback mechanism for empty device selection
    if not device_types:
        st.warning("⚠️ Please select at least one device type!")
        # Add an empty space to make the warning more visible
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("Use the sidebar to select device types to display performance metrics.")
        # Skip the rest of the content by adding an early return
        st.stop()

    # Add metric explanation in sidebar expander
    with st.sidebar.expander("Metric Explanations"):
        st.markdown("""
        **TECA (Total Energy Correctly Assigned):** This metric evaluates the core technical function of NILM algorithms: accurate energy decomposition. It measures not just classification accuracy but quantification precision, providing a direct measure of algorithm performance against the fundamental technical objective of energy disaggregation. This metric is reproducible across different datasets and allows direct comparison with other NILM research.
        
        **DPSPerc (%):** Selected for its statistical robustness in handling class imbalance, which is inherent in NILM applications where device usage events are typically sparse in the overall energy signal. DPSPerc provides a balanced evaluation by giving equal weight to both successful detection and non-detection, making it more reliable than accuracy across varying device usage frequencies. We use the recall FPP plane in this case.
        
        **FPR (False Positive Rate):** Essential for technical validation as it directly quantifies Type I errors in our statistical model. In signal processing applications like NILM, false positives introduce systematic bias in energy disaggregation, affecting model reliability more severely than false negatives. This metric is especially critical when evaluating algorithm performance on noisy energy signals with multiple overlapping loads.
        
        """)

    # Add technical motivation in sidebar expander
    with st.sidebar.expander("Why These Metrics?"):
        st.markdown("""
        **Technical Rationale:**
        
        • **DPSPerc:** Statistically robust to class imbalance inherent in NILM applications where device usage events are sparse in the overall energy signal.
        
        • **FPR:** Directly quantifies Type I errors, crucial for NILM where false positives introduce systematic bias in energy disaggregation.
        
        • **TECA:** Evaluates the core technical function of NILM: accurate energy decomposition, providing direct measure of algorithm performance.
        
        These metrics maintain consistency across households with different energy consumption patterns, making model evaluation more reproducible and generalizable.
        """)

    st.sidebar.markdown("---")
    st.sidebar.info("This dashboard presents the performance metrics of our NILM algorithm for detecting various device usage patterns.")

    # After the sidebar elements but before the Key metrics display
    st.markdown(f"""
    This dashboard presents performance metrics for the **{selected_model} ({model_dates[selected_model]})** model 
    in detecting EV Charging, AC Usage, PV Usage, and WH Usage consumption patterns. For this benchmark, we have used six versions of the model (V1-V6), each with different hyperparameters and training strategies.
    """)

    # Main content area for Performance Metrics page
    st.subheader(f"Model: {selected_model} ({model_dates[selected_model]})" )

    # Key metrics display section - now in a 2-column layout
    metrics_col, trend_col = st.columns([1, 1])
    
    with metrics_col:
        st.markdown("### Key Metrics")

        # Create metrics in 4-column layout for the selected model
        metrics_cols = st.columns(min(4, len(device_types)))

        # Define colors for each device
        device_colors = {
            'EV Charging': primary_purple,
            'AC Usage': green,
            'PV Usage': cream,
            'WH Usage': salmon
        }

        # Display metric cards for each device type
        for i, device in enumerate(device_types):
            with metrics_cols[i % len(metrics_cols)]:
                st.markdown(f"<h4 style='color:{device_colors[device]};'>{device}</h4>", unsafe_allow_html=True)
                
                # DPSPerc
                st.metric(
                    "DPSPerc (%)", 
                    f"{models_data[selected_model]['DPSPerc'][device]:.2f}%",
                    delta=None,
                    delta_color="normal"
                )
                
                # FPR - Lower is better, so we'll display the inverse
                fpr_value = models_data[selected_model]['FPR'][device]
                st.metric(
                    "False Positive Rate", 
                    f"{fpr_value * 100:.2f}%",
                    delta=None,
                    delta_color="normal"
                )
                
                # TECA - convert to percentage
                teca_value = models_data[selected_model]['TECA'][device]
                st.metric(
                    "TECA (%)", 
                    f"{teca_value * 100:.2f}%",
                    delta=None,
                    delta_color="normal"
                )

    with trend_col:
        st.markdown("### Performance Trend")
        
        # Get the first device as default if device_types is not empty
        default_device = device_types[0] if device_types else "EV Charging"
        
        # Allow user to select a device for the trend visualization
        trend_device = st.selectbox(
            "Select device for trend analysis", 
            device_types,
            index=0
        )
        
        # Prepare data for the line chart showing performance over model versions
        trend_data = []
        
        for model in ["V1", "V2", "V3", "V4", "V5", "V6"]:
            trend_data.append({
                "Model": f"{model} ({model_dates[model]})",
                "DPSPerc (%)": models_data[model]["DPSPerc"][trend_device],
                "FPR (%)": models_data[model]["FPR"][trend_device] * 100,  # Convert to percentage
                "TECA (%)": models_data[model]["TECA"][trend_device] * 100  # Convert to percentage
            })
        
        trend_df = pd.DataFrame(trend_data)
        
        # Create a mapping to convert display names back to model keys
        model_display_to_key = {f"{model} ({model_dates[model]})": model for model in ["V1", "V2", "V3", "V4", "V5", "V6"]}
        
        # Find best model for each metric
        best_dpsperc_display = trend_df.loc[trend_df["DPSPerc (%)"].idxmax()]["Model"]
        best_fpr_display = trend_df.loc[trend_df["FPR (%)"].idxmin()]["Model"]  # Lowest is best for FPR
        best_teca_display = trend_df.loc[trend_df["TECA (%)"].idxmax()]["Model"]
        
        # Store original model keys for data lookup
        best_dpsperc_model = model_display_to_key[best_dpsperc_display]
        best_fpr_model = model_display_to_key[best_fpr_display]
        best_teca_model = model_display_to_key[best_teca_display]
        
        # Create line chart
        fig = go.Figure()
        
        # Add DPSPerc line
        fig.add_trace(go.Scatter(
            x=trend_df["Model"],
            y=trend_df["DPSPerc (%)"],
            mode='lines+markers',
            name='DPSPerc (%)',
            line=dict(color=primary_purple, width=3),
            marker=dict(size=10)
        ))
        
        # Add FPR line
        fig.add_trace(go.Scatter(
            x=trend_df["Model"],
            y=trend_df["FPR (%)"],
            mode='lines+markers',
            name='FPR (%)',
            line=dict(color=salmon, width=3),
            marker=dict(size=10)
        ))
        
        # Add TECA line
        fig.add_trace(go.Scatter(
            x=trend_df["Model"],
            y=trend_df["TECA (%)"],
            mode='lines+markers',
            name='TECA (%)',
            line=dict(color=green, width=3),
            marker=dict(size=10)
        ))
        
        # Highlight best model for each metric with stars
        # DPSPerc best
        dpsperc_best_idx = trend_df[trend_df["Model"] == best_dpsperc_display].index[0]
        fig.add_trace(go.Scatter(
            x=[best_dpsperc_display],
            y=[trend_df.iloc[dpsperc_best_idx]["DPSPerc (%)"]],
            mode='markers',
            marker=dict(
                symbol='star',
                size=16,
                color=primary_purple,
                line=dict(color='white', width=1)
            ),
            name="Best DPSPerc",
            showlegend=True
        ))
        
        # FPR best
        fpr_best_idx = trend_df[trend_df["Model"] == best_fpr_display].index[0]
        fig.add_trace(go.Scatter(
            x=[best_fpr_display],
            y=[trend_df.iloc[fpr_best_idx]["FPR (%)"]],
            mode='markers',
            marker=dict(
                symbol='star',
                size=16,
                color=salmon,
                line=dict(color='white', width=1)
            ),
            name="Best FPR",
            showlegend=True
        ))
        
        # TECA best
        teca_best_idx = trend_df[trend_df["Model"] == best_teca_display].index[0]
        fig.add_trace(go.Scatter(
            x=[best_teca_display],
            y=[trend_df.iloc[teca_best_idx]["TECA (%)"]],
            mode='markers',
            marker=dict(
                symbol='star',
                size=16,
                color=green,
                line=dict(color='white', width=1)
            ),
            name="Best TECA",
            showlegend=True
        ))
        
        # Define the selected model display name here
        selected_model_display = f"{selected_model} ({model_dates[selected_model]})"
        
        # Highlight the currently selected model with circles
        current_model_index = ["V1", "V2", "V3", "V4", "V5", "V6"].index(selected_model)
        
        # Add vertical line for currently selected model
        fig.add_shape(
            type="line",
            x0=selected_model_display,
            y0=0,
            x1=selected_model_display,
            y1=100,
            line=dict(
                color="rgba(80, 80, 80, 0.3)",
                width=6,
                dash="dot",
            )
        )
        
        # Add timeline elements - faint vertical lines and model version labels at bottom
        model_display_names = [f"{model} ({model_dates[model]})" for model in ["V1", "V2", "V3", "V4", "V5", "V6"]]
        
        for model_display in model_display_names:
            if model_display != selected_model_display:  # We already added a line for the selected model
                fig.add_shape(
                    type="line",
                    x0=model_display,
                    y0=0,
                    x1=model_display,
                    y1=100,
                    line=dict(
                        color="rgba(200, 200, 200, 0.3)",
                        width=1,
                    )
                )
        
        # Determine which model has best average performance
        trend_df['FPR_inv'] = 100 - trend_df['FPR (%)']  # Invert FPR so higher is better
        trend_df['avg_score'] = (trend_df['DPSPerc (%)'] + trend_df['FPR_inv'] + trend_df['TECA (%)']) / 3
        best_overall_display = trend_df.loc[trend_df['avg_score'].idxmax()]["Model"]
        best_overall = model_display_to_key[best_overall_display]

        # Add highlight for the best overall model - using a vertical bar instead of rectangle
        fig.add_shape(
            type="line",
            x0=best_overall_display,
            y0=0,
            x1=best_overall_display,
            y1=100,
            line=dict(
                color=primary_purple,
                width=10,
            ),
            opacity=0.3,
            layer="below",
        )
        
        # Add "Best Overall" annotation for models
        if len(set([best_dpsperc_model, best_fpr_model, best_teca_model])) == 1:
            # If one model is best at everything
            best_model = best_dpsperc_model
            best_model_display = best_dpsperc_display
            fig.add_annotation(
                x=best_model_display,
                y=100,
                text=f"BEST OVERALL",
                showarrow=False,
                yanchor="bottom",
                font=dict(color=dark_purple, size=12, family="Arial Black"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=primary_purple,
                borderwidth=2,
                borderpad=4
            )
        else:            
            fig.add_annotation(
                x=best_overall_display,
                y=100,
                text=f"BEST OVERALL",
                showarrow=False,
                yanchor="bottom",
                font=dict(color=dark_purple, size=12, family="Arial Black"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=primary_purple,
                borderwidth=2,
                borderpad=4
            )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Model Version Timeline",
            yaxis_title="Performance (%)",
            xaxis=dict(
                tickmode='array',
                tickvals=trend_df["Model"].tolist(),
                ticktext=trend_df["Model"].tolist(),
                tickangle=0,
                tickfont=dict(size=12, color=dark_purple),
                showgrid=False
            ),
            yaxis=dict(
                # Use full range to show all metrics including FPR
                range=[0, 100]
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            paper_bgcolor=white,
            plot_bgcolor=white,
            font=dict(color=dark_purple),
            height=500,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        # Add a note about FPR
        fig.add_annotation(
            x=0.98,
            y=0.03,
            xref="paper",
            yref="paper",
            text="Note: Lower FPR is better",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor=salmon,
            borderwidth=1,
            borderpad=4,
            font=dict(color=dark_purple, size=10)
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Add key stats about best models
        st.markdown(f"""
        <div style="border-left: 3px solid {primary_purple}; padding-left: 10px; margin: 10px 0;">
            <small>
            <strong>Best DPSPerc:</strong> {best_dpsperc_display} ({trend_df.loc[trend_df['Model'] == best_dpsperc_display, 'DPSPerc (%)'].values[0]:.2f}%)<br>
            <strong>Best FPR:</strong> {best_fpr_display} ({trend_df.loc[trend_df['Model'] == best_fpr_display, 'FPR (%)'].values[0]:.2f}%)<br>
            <strong>Best TECA:</strong> {best_teca_display} ({trend_df.loc[trend_df['Model'] == best_teca_display, 'TECA (%)'].values[0]:.2f}%)
            </small>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a brief explanation
        if selected_model == "V6":
            st.caption(f"The timeline shows how metrics for {trend_device} have evolved through model versions. Stars indicate best performance for each metric.")
        else:
            st.caption(f"The timeline shows how metrics for {trend_device} have evolved through versions. Try selecting different models to compare their performance.")

    # Sample Composition Table
    st.markdown("### Sample Composition")
    st.markdown("Distribution of positive and negative samples in the test dataset:")

    # Determine which sample composition to show based on selected model
    if selected_model == "V6":
        # New sample composition for V6
        sample_composition = {
            'Device Type': ['EV Charging', 'AC Usage', 'PV Usage'],
            'Negative Class (%)': [77.08, 33.33, 31.25],
            'Positive Class (%)': [22.92, 66.67, 68.75]
        }
        
        # Create a DataFrame
        sample_df = pd.DataFrame(sample_composition)
        
        # Add count information in tooltips
        sample_df['Negative Count'] = [37, 16, 15]
        sample_df['Positive Count'] = [11, 32, 33]
    else:
        # Original sample composition for V1-V5
        sample_composition = {
            'Device Type': ['EV Charging', 'AC Usage', 'PV Usage'],
            'Negative Class (%)': [64.56, 54.43, 36.08],
            'Positive Class (%)': [35.44, 45.57, 63.92]
        }
        
        # Create a DataFrame
        sample_df = pd.DataFrame(sample_composition)
        
        # Add count information in tooltips
        sample_df['Negative Count'] = [102, 86, 57]
        sample_df['Positive Count'] = [56, 72, 101]

    # Style the table
    def highlight_class_imbalance(val):
        """Highlight cells based on class balance"""
        if isinstance(val, float):
            if val < 20:
                return 'background-color: rgba(211, 116, 95, 0.2)'  # Light red for very imbalanced
            elif val > 80:
                return 'background-color: rgba(211, 116, 95, 0.2)'  # Light red for very imbalanced
        return ''

    # Apply styling with custom formatting for percentages
    styled_df = sample_df[['Device Type', 'Negative Class (%)', 'Positive Class (%)']].style\
        .format({'Negative Class (%)': '{:.2f}%', 'Positive Class (%)': '{:.2f}%'})\
        .applymap(highlight_class_imbalance, subset=['Negative Class (%)', 'Positive Class (%)']) \
        .set_properties(**{'text-align': 'center', 'font-size': '1rem', 'border-color': light_purple})

    st.table(styled_df)

    # Add an explanatory note about class imbalance
    st.caption(f"""
    Sample composition shows the distribution of positive and negative examples in our test dataset.
    {
    "The V6 model was tested on a different dataset with more EV charging negative examples (77.1% negative class) and more AC and PV positive examples (66.7% and 68.8% positive class respectively)." 
    if selected_model == "V6" else 
    "The dataset has a relatively balanced distribution for AC and EV detection, while PV detection has more positive examples."
    }
    """)

    # Model comparison
    st.markdown("### Model Comparison")

    # Create tabs for different metrics
    metric_tabs = st.tabs(["DPSPerc", "FPR", "TECA"])

    with metric_tabs[0]:  # DPSPerc tab
        # Create bar chart for DPSPerc across models and devices
        dpsperc_data = []
        for model in ["V1", "V2", "V3", "V4", "V5", "V6"]:
            for device in device_types:
                dpsperc_data.append({
                    "Model": f"{model} ({model_dates[model]})",
                    "Device": device,
                    "DPSPerc (%)": models_data[model]["DPSPerc"][device]
                })
        
        dpsperc_df = pd.DataFrame(dpsperc_data)
        
        fig = px.bar(
            dpsperc_df, 
            x="Model", 
            y="DPSPerc (%)", 
            color="Device",
            barmode="group",
            color_discrete_map={
                "EV Charging": primary_purple,
                "AC Usage": green,
                "PV Usage": cream
            },
            template="plotly_white"
        )
        
        fig.update_layout(
            title="DPSPerc Comparison Across Models (Higher is Better)",
            xaxis_title="Model",
            yaxis_title="DPSPerc (%)",
            legend_title="Device",
            paper_bgcolor=white,
            plot_bgcolor=white,
            font=dict(color=dark_purple),
            yaxis=dict(
                # Set a minimum value to better highlight differences
                range=[70, 100]
            )
        )
        
        # Add a note about the zoomed y-axis
        fig.add_annotation(
            x=0.02,
            y=0.03,
            xref="paper",
            yref="paper",
            text="Note: Y-axis starts at 70% to highlight differences",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor=dark_purple,
            borderwidth=1,
            borderpad=4,
            font=dict(color=dark_purple, size=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with metric_tabs[1]:  # FPR tab
        # Create bar chart for FPR across models and devices
        fpr_data = []
        for model in ["V1", "V2", "V3", "V4", "V5", "V6"]:
            for device in device_types:
                fpr_data.append({
                    "Model": f"{model} ({model_dates[model]})",
                    "Device": device,
                    "FPR (%)": models_data[model]["FPR"][device] * 100  # Convert to percentage
                })
        
        fpr_df = pd.DataFrame(fpr_data)
        
        fig = px.bar(
            fpr_df, 
            x="Model", 
            y="FPR (%)", 
            color="Device",
            barmode="group",
            color_discrete_map={
                "EV Charging": primary_purple,
                "AC Usage": green,
                "PV Usage": cream
            },
            template="plotly_white"
        )
        
        fig.update_layout(
            title="False Positive Rate Comparison Across Models (Lower is Better)",
            xaxis_title="Model",
            yaxis_title="FPR (%)",
            legend_title="Device",
            paper_bgcolor=white,
            plot_bgcolor=white,
            font=dict(color=dark_purple),
            yaxis=dict(
                # Set a maximum value to better highlight differences
                range=[0, 50]
            )
        )
        
        # Add a note about the zoomed y-axis
        fig.add_annotation(
            x=0.02,
            y=0.97,
            xref="paper",
            yref="paper",
            text="Note: Y-axis limited to 50% for better visibility",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor=dark_purple,
            borderwidth=1,
            borderpad=4,
            font=dict(color=dark_purple, size=10)
        )

        # Add disclaimer about V6 FPR bias
        fig.add_annotation(
            x=0.98,
            y=0.03,
            xref="paper",
            yref="paper",
            text="Disclaimer: V6 model FPR is biased towards false negatives",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor=red,
            borderwidth=1,
            borderpad=4,
            font=dict(color=dark_purple, size=10),
            align="right"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with metric_tabs[2]:  # TECA tab
        # Create bar chart for TECA across models and devices
        teca_data = []
        for model in ["V1", "V2", "V3", "V4", "V5", "V6"]:
            for device in device_types:
                teca_data.append({
                    "Model": f"{model} ({model_dates[model]})",
                    "Device": device,
                    "TECA (%)": models_data[model]["TECA"][device] * 100
                })
        
        teca_df = pd.DataFrame(teca_data)
        
        fig = px.bar(
            teca_df, 
            x="Model", 
            y="TECA (%)", 
            color="Device",
            barmode="group",
            color_discrete_map={
                "EV Charging": primary_purple,
                "AC Usage": green,
                "PV Usage": cream
            },
            template="plotly_white"
        )
        
        fig.update_layout(
            title="Total Energy Correctly Assigned Across Models (Higher is Better)",
            xaxis_title="Model",
            yaxis_title="TECA (%)",
            legend_title="Device",
            paper_bgcolor=white,
            plot_bgcolor=white,
            font=dict(color=dark_purple),
            yaxis=dict(
                # Set a minimum value to better highlight differences
                range=[60, 100]
            )
        )
        
        # Add a note about the zoomed y-axis
        fig.add_annotation(
            x=0.02,
            y=0.03,
            xref="paper",
            yref="paper",
            text="Note: Y-axis starts at 60% to highlight differences",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor=dark_purple,
            borderwidth=1,
            borderpad=4,
            font=dict(color=dark_purple, size=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # Create two columns for radar plot and confusion matrix
    st.markdown("### Performance Visualization")
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        # Radar Chart
        categories = ['DPSPerc', 'Low FPR', 'TECA']
        
        # Process data for radar chart (0-1)
        radar_data = {}
        for device in device_types:
            radar_data[device] = [
                models_data[selected_model]['DPSPerc'][device]/100,  # DPSPerc scaled to 0-1
                1 - models_data[selected_model]['FPR'][device],      # Invert FPR (higher is better)
                max(0, models_data[selected_model]['TECA'][device])  # TECA already in 0-1 range
            ]

        # Create radar chart
        fig = go.Figure()

        for device in device_types:
            fig.add_trace(go.Scatterpolar(
                r=radar_data[device] + [radar_data[device][0]],  # Close the loop
                theta=categories + [categories[0]],
                fill='toself',
                name=device,
                line_color=device_colors[device]
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            paper_bgcolor=white,
            plot_bgcolor=white,
            font=dict(color=dark_purple),
            margin=dict(l=20, r=20, t=30, b=20),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Performance metrics for {selected_model} ({model_dates[selected_model]}) model across all dimensions")

    with viz_col2:
        # Add device selector for confusion matrix
        selected_device = st.selectbox(
            "Select Device for Confusion Matrix",
            device_types,
            key="confusion_matrix_selector"
        )
        
        # Create synthetic confusion matrix based on DPSPerc and FPR
        # This is an approximation since we don't have the actual confusion matrix values
        dpsperc = models_data[selected_model]['DPSPerc'][selected_device]
        fpr = models_data[selected_model]['FPR'][selected_device]
        
        # Calculate true positive rate (TPR) from DPSPerc
        tpr = dpsperc / 100
        
        # Calculate true negative rate (TNR) from FPR
        tnr = 1 - fpr
        
        # Create confusion matrix
        cm = np.array([
            [tnr*100, fpr*100],         # True Negatives %, False Positives %
            [(1-tpr)*100, tpr*100]      # False Negatives %, True Positives %
        ])
        
        # Determine labels based on device
        if selected_device == "EV Charging":
            labels = ['No EV', 'EV Detected']
            title = 'EV Charging Detection'
            cmap = 'Blues'
        elif selected_device == "AC Usage":
            labels = ['No AC', 'AC Detected']
            title = 'AC Usage Detection'
            cmap = 'Greens'
        else:  # PV Usage
            labels = ['No PV', 'PV Detected']
            title = 'PV Usage Detection'
            cmap = 'Oranges'

        # Create and display the selected confusion matrix
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='.1f', cmap=cmap, ax=ax,
                  xticklabels=labels,
                  yticklabels=[f'No {selected_device.split()[0]}', f'{selected_device.split()[0]} Present'])
        plt.title(title)
        st.pyplot(fig)
        st.caption(f"Confusion matrix for {selected_device} detection (%) with {selected_model} ({model_dates[selected_model]}) model")

    # Key findings section
    st.markdown("### Key Findings")
    st.markdown(f"""
    - The **{selected_model} ({model_dates[selected_model]})** model shows strong performance across all device types, with PV Usage detection being particularly accurate.
    - EV Charging detection shows a good balance between DPSPerc ({models_data[selected_model]['DPSPerc']['EV Charging']:.2f}%) and false positives ({models_data[selected_model]['FPR']['EV Charging'] * 100:.2f}%).
    - PV Usage detection achieves the highest DPSPerc ({models_data[selected_model]['DPSPerc']['PV Usage']:.2f}%) among all device types.
    """)

    # Add model comparison insights
    if selected_model == "V4":
        st.markdown("""
        **Model Comparison Insights:**
        - The V4 (March 13) model achieves the highest EV Charging DPSPerc (85.81%) with the lowest FPR (11.76%), offering the most accurate EV detection.
        - AC Usage detection shows comparable performance across models, with the V4 model providing the best balance of accuracy and false positives.
        - All models demonstrate similar PV detection capabilities, with minor variations in performance metrics.
        """)
    elif selected_model == "V5":
        st.markdown("""
        **Model Comparison Insights:**
        - The V5 (March 18) model offers improved AC Usage detection with a DPSPerc of 77.58%, higher than previous models.
        - For EV Charging, V5 provides a balanced approach with a DPSPerc of 81.11% and moderate FPR of 13.73%, making it more reliable than earlier versions but not as aggressive as V4 (March 13).
        - PV Usage detection in V5 (91.93% DPSPerc) remains strong and consistent with previous models.
        - The TECA scores show that V5 achieves good energy assignment accuracy for AC Usage (0.6912) and PV Usage (0.7680), though slightly lower for EV Charging (0.6697) compared to V4.
        - Overall, V5 represents a more balanced model that prioritizes consistent performance across all device types rather than optimizing for any single metric.
        """)
    elif selected_model == "V6":
        st.markdown("""
        **Model Comparison Insights:**
        - The V6 (March 25) model represents a significant breakthrough in overall performance balance across all device types.
        - EV Charging detection shows the highest DPSPerc (86.58%) among all models with an impressively low FPR of just 5.41%.
        - AC Usage detection achieves a notable improvement with a DPSPerc of 83.05%, substantially higher than any previous model.
        - PV Usage detection achieves a perfect 0% false positive rate while maintaining a strong DPSPerc of 85.00%.
        - TECA scores are well-balanced, with particularly strong energy assignment accuracy for AC Usage (0.7739).
        - The V6 model demonstrates remarkable improvement in reducing false positives while maintaining or improving detection accuracy.
        - The model's performance is particularly impressive considering the different sample distribution compared to earlier versions.
        """)

    # Footer (using the same styling as the main page)
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; color:{primary_purple}; padding: 10px; border-radius: 5px;">
        This dashboard presents the performance metrics of our NILM algorithm for detecting various device usage patterns.
        These results help guide model selection and identify areas for further improvement.
    </div>
    """, unsafe_allow_html=True)

elif page == "Interactive Map":
    # Interactive Map page
    st.title("NILM Deployment Map")
    st.subheader("Geographic Distribution of Homes with Smart Devices")
    
    # Description
    st.markdown("""
    This map shows the geographic distribution of homes equipped with NILM-detected devices.
    **Use the dropdown below** to select a state and view individual homes with EV chargers, AC units, and solar panels.
    """)
    
    # --- Load Real Household Data ---
    @st.cache_data
    def load_household_data(csv_path='customer_device_mapping_with_coords.csv'):
        try:
            df = pd.read_csv(csv_path)
            # Rename columns for consistency
            rename_map = {
                'custid': 'id', 'latitude': 'lat', 'longitude': 'lon', 'state': 'state'
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            # Check for essential columns
            required_core_cols = ['id', 'lat', 'lon', 'state']
            value_cols = ['ac_value', 'pv_value', 'ev_value']
            missing_core_cols = [col for col in required_core_cols if col not in df.columns]
            if missing_core_cols:
                st.error(f"Error: CSV '{csv_path}' is missing required core columns: {missing_core_cols}")
                return pd.DataFrame()
            # Ensure value columns exist, fill missing with 0
            for val_col in value_cols:
                if val_col not in df.columns:
                    st.warning(f"Warning: Column '{val_col}' not found in CSV. Assuming 0 for all entries.")
                    df[val_col] = 0
                else:
                    df[val_col] = pd.to_numeric(df[val_col], errors='coerce').fillna(0)
            # Create boolean flags for device presence
            df['has_ac'] = df['ac_value'] > 0
            df['has_pv'] = df['pv_value'] > 0
            df['has_ev'] = df['ev_value'] > 0
            # Select and return relevant columns
            final_cols = ['id', 'lat', 'lon', 'state', 'has_ev', 'has_ac', 'has_pv']
            return df[[col for col in final_cols if col in df.columns]]
        except FileNotFoundError:
            st.error(f"Error: Household data file '{csv_path}' not found.")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading or processing household data from '{csv_path}': {e}")
            return pd.DataFrame()

    # --- Refactored Data Generation ---
    @st.cache_data
    def get_states_info():
        # ... (States data dictionary) ...
        states_data = {
            'AL': {'name': 'Alabama', 'lat': 32.7794, 'lon': -86.8287, 'zoom': 7},
            'AK': {'name': 'Alaska', 'lat': 64.0685, 'lon': -152.2782, 'zoom': 4},
            'AZ': {'name': 'Arizona', 'lat': 34.2744, 'lon': -111.6602, 'zoom': 7},
            'AR': {'name': 'Arkansas', 'lat': 34.8938, 'lon': -92.4426, 'zoom': 7},
            'CA': {'name': 'California', 'lat': 36.7783, 'lon': -119.4179, 'zoom': 6},
            'CO': {'name': 'Colorado', 'lat': 39.5501, 'lon': -105.7821, 'zoom': 7},
            'CT': {'name': 'Connecticut', 'lat': 41.6219, 'lon': -72.7273, 'zoom': 8},
            'DE': {'name': 'Delaware', 'lat': 38.9896, 'lon': -75.5050, 'zoom': 8},
            'DC': {'name': 'District of Columbia', 'lat': 38.9072, 'lon': -77.0369, 'zoom': 10},
            'FL': {'name': 'Florida', 'lat': 27.6648, 'lon': -81.5158, 'zoom': 6},
            'GA': {'name': 'Georgia', 'lat': 32.6415, 'lon': -83.4426, 'zoom': 7},
            'HI': {'name': 'Hawaii', 'lat': 20.2927, 'lon': -156.3737, 'zoom': 7},
            'ID': {'name': 'Idaho', 'lat': 44.0682, 'lon': -114.7420, 'zoom': 6},
            'IL': {'name': 'Illinois', 'lat': 40.0417, 'lon': -89.1965, 'zoom': 7},
            'IN': {'name': 'Indiana', 'lat': 39.8942, 'lon': -86.2816, 'zoom': 7},
            'IA': {'name': 'Iowa', 'lat': 42.0751, 'lon': -93.4960, 'zoom': 7},
            'KS': {'name': 'Kansas', 'lat': 38.4937, 'lon': -98.3804, 'zoom': 7},
            'KY': {'name': 'Kentucky', 'lat': 37.5347, 'lon': -85.3021, 'zoom': 7},
            'LA': {'name': 'Louisiana', 'lat': 31.0689, 'lon': -91.9968, 'zoom': 7},
            'ME': {'name': 'Maine', 'lat': 45.3695, 'lon': -69.2428, 'zoom': 7},
            'MD': {'name': 'Maryland', 'lat': 39.0550, 'lon': -76.7909, 'zoom': 7},
            'MA': {'name': 'Massachusetts', 'lat': 42.2596, 'lon': -71.8083, 'zoom': 8},
            'MI': {'name': 'Michigan', 'lat': 44.3467, 'lon': -85.4102, 'zoom': 7},
            'MN': {'name': 'Minnesota', 'lat': 46.2807, 'lon': -94.3053, 'zoom': 6},
            'MS': {'name': 'Mississippi', 'lat': 32.7364, 'lon': -89.6678, 'zoom': 7},
            'MO': {'name': 'Missouri', 'lat': 38.3566, 'lon': -92.4580, 'zoom': 7},
            'MT': {'name': 'Montana', 'lat': 46.8797, 'lon': -110.3626, 'zoom': 6},
            'NE': {'name': 'Nebraska', 'lat': 41.5378, 'lon': -99.7951, 'zoom': 7},
            'NV': {'name': 'Nevada', 'lat': 39.3289, 'lon': -116.6312, 'zoom': 6},
            'NH': {'name': 'New Hampshire', 'lat': 43.6805, 'lon': -71.5811, 'zoom': 7},
            'NJ': {'name': 'New Jersey', 'lat': 40.1907, 'lon': -74.6728, 'zoom': 7},
            'NM': {'name': 'New Mexico', 'lat': 34.4071, 'lon': -106.1126, 'zoom': 6},
            'NY': {'name': 'New York', 'lat': 42.9538, 'lon': -75.5268, 'zoom': 7},
            'NC': {'name': 'North Carolina', 'lat': 35.5557, 'lon': -79.3877, 'zoom': 7},
            'ND': {'name': 'North Dakota', 'lat': 47.4501, 'lon': -100.4659, 'zoom': 7},
            'OH': {'name': 'Ohio', 'lat': 40.2862, 'lon': -82.7937, 'zoom': 7},
            'OK': {'name': 'Oklahoma', 'lat': 35.5889, 'lon': -97.4943, 'zoom': 7},
            'OR': {'name': 'Oregon', 'lat': 43.9336, 'lon': -120.5583, 'zoom': 7},
            'PA': {'name': 'Pennsylvania', 'lat': 40.8781, 'lon': -77.7996, 'zoom': 7},
            'RI': {'name': 'Rhode Island', 'lat': 41.6762, 'lon': -71.5562, 'zoom': 9},
            'SC': {'name': 'South Carolina', 'lat': 33.9169, 'lon': -80.8964, 'zoom': 7},
            'SD': {'name': 'South Dakota', 'lat': 44.4443, 'lon': -100.2263, 'zoom': 7},
            'TN': {'name': 'Tennessee', 'lat': 35.8580, 'lon': -86.3505, 'zoom': 7},
            'TX': {'name': 'Texas', 'lat': 31.4757, 'lon': -99.3312, 'zoom': 6},
            'UT': {'name': 'Utah', 'lat': 39.3055, 'lon': -111.6703, 'zoom': 7},
            'VT': {'name': 'Vermont', 'lat': 44.0687, 'lon': -72.6658, 'zoom': 7},
            'VA': {'name': 'Virginia', 'lat': 37.5215, 'lon': -78.8537, 'zoom': 7},
            'WA': {'name': 'Washington', 'lat': 47.3826, 'lon': -120.4472, 'zoom': 7},
            'WV': {'name': 'West Virginia', 'lat': 38.6409, 'lon': -80.6227, 'zoom': 7},
            'WI': {'name': 'Wisconsin', 'lat': 44.6243, 'lon': -89.9941, 'zoom': 7},
            'WY': {'name': 'Wyoming', 'lat': 42.9957, 'lon': -107.5512, 'zoom': 7}
        }
        return states_data

    @st.cache_data
    def load_us_geojson():
        try:
            response = requests.get("https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json")
            response.raise_for_status() 
            us_states = response.json()
            return us_states
        except requests.exceptions.RequestException as e:
            st.error(f"Error loading US state boundaries: {e}")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred while processing GeoJSON: {e}")
            return None

    # Load data
    states_info = get_states_info()
    us_geojson = load_us_geojson()
    all_household_data = load_household_data()

    # --- Sidebar Elements --- 
    st.sidebar.markdown("### Map Filters")
    show_ev = st.sidebar.checkbox("Show EV Chargers", value=True)
    show_ac = st.sidebar.checkbox("Show AC Units", value=True)
    show_pv = st.sidebar.checkbox("Show Solar Panels", value=True)
    st.sidebar.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
    show_all_three = st.sidebar.checkbox("ONLY Show Homes with All 3 Devices", value=False)
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <style>
    div[data-testid="stButton"] button {
        background-color: #515D9A !important;
        color: white !important;
        font-weight: bold !important;
        border: 1px solid darkgray !important;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #B8BCF3 !important;
        color: #202842 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    if st.sidebar.button("↻ Reset Map View", use_container_width=True):
        if 'selected_state_map' in st.session_state:
            del st.session_state['selected_state_map']
        st.rerun()
    # --- End Sidebar Elements ---

    # --- State Selection --- 
    if not all_household_data.empty and 'state' in all_household_data.columns:
        valid_states = all_household_data['state'].dropna().astype(str).str.upper().unique()
        available_states = sorted([s for s in valid_states if s != 'NA'])
        if not available_states and 'NA' in valid_states:
             st.warning("Only 'NA' found for state values in the data.")
             available_states = []
        elif 'NA' in valid_states:
             st.info("Ignoring entries with state 'NA' in the dropdown.")
        state_names = {code: states_info.get(code, {}).get('name', code) for code in available_states}
        sorted_state_options = sorted(state_names.items(), key=lambda item: item[1])
        state_options = ["Select a State..."] + [name for code, name in sorted_state_options]
        state_code_map = {name: code for code, name in sorted_state_options}
    else:
        st.warning("Using default state list as household data is unavailable or missing/empty 'state' column.")
        state_names = {code: info['name'] for code, info in states_info.items()}
        state_codes_sorted = sorted(state_names.keys(), key=lambda k: state_names[k])
        state_options = ["Select a State..."] + [state_names[code] for code in state_codes_sorted]
        # Corrected indentation for state_code_map
        state_code_map = {v: k for k, v in state_names.items()} # Map name back to code

    if 'selected_state_map' not in st.session_state or st.session_state['selected_state_map'] not in state_options:
        st.session_state['selected_state_map'] = state_options[0]

    selected_state_name = st.selectbox(
        "Select a State to View Households:",
        options=state_options,
        key='selected_state_map',
        index=state_options.index(st.session_state['selected_state_map'])
    )

    selected_state_code = None
    if selected_state_name != "Select a State...":
        selected_state_code = state_code_map.get(selected_state_name)

    # --- Create Map --- 
    map_location = [39.8283, -98.5795]
    map_zoom = 4
    if selected_state_code and selected_state_code in states_info:
        state_info = states_info[selected_state_code]
        map_location = [state_info['lat'], state_info['lon']]
        map_zoom = state_info['zoom']
    elif selected_state_code and selected_state_code not in states_info:
         st.warning(f"State code '{selected_state_code}' found in data but not in state center/zoom lookup. Using default map view.")

    m = folium.Map(
        location=map_location, 
        zoom_start=map_zoom,
        tiles="CartoDB positron" 
    )
    folium.TileLayer('Esri_WorldImagery', name='Satellite View', attr='Esri').add_to(m)
    
    if us_geojson:
        style_function = lambda x: {
            'fillColor': primary_purple,
            'color': 'white',
            'weight': 1,
            'fillOpacity': 0.3 if selected_state_code else 0.5
        }
        highlight_function = lambda x: {
                'fillColor': light_purple,
                'color': 'white',
                'weight': 3,
                'fillOpacity': 0.7
        } if not selected_state_code else style_function(x)

        folium.GeoJson(
            us_geojson,
            name='State Boundaries',
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['State:'], labels=True, sticky=False),
        ).add_to(m)

    # --- Load and Add Household Markers --- 
    filtered_households_df = pd.DataFrame()
    if selected_state_code and not all_household_data.empty:
        with st.spinner(f"Filtering household data for {selected_state_name}..."):
            state_households_df = all_household_data[all_household_data['state'].astype(str).str.upper() == selected_state_code]
            if show_all_three:
                all_three_condition = state_households_df['has_ev'] & state_households_df['has_ac'] & state_households_df['has_pv']
                filtered_households_df = state_households_df[all_three_condition]
                st.info("Filter applied: Showing only homes with all three devices (EV, AC, PV).")
            else:
                filters = []
                if show_ev: filters.append(state_households_df['has_ev'])
                if show_ac: filters.append(state_households_df['has_ac'])
                if show_pv: filters.append(state_households_df['has_pv'])
                if filters:
                    combined_filter = pd.DataFrame(filters).transpose().any(axis=1)
                    filtered_households_df = state_households_df[combined_filter]
                elif not state_households_df.empty:
                    filtered_households_df = pd.DataFrame(columns=state_households_df.columns)
                    st.info(f"No device filters selected. Please select device types or 'All 3 Devices' to view markers.")

        if not filtered_households_df.empty:
            marker_cluster = MarkerCluster(name=f'{selected_state_name} Homes').add_to(m)
            for idx, house in filtered_households_df.iterrows():
                device_count = int(house['has_ev']) + int(house['has_ac']) + int(house['has_pv'])
                
                # Assign icon based on device count and filters - Corrected indentation
                if device_count == 3:
                    icon_color = 'purple'
                    icon_name = 'star'
                    tooltip_device = 'All Devices (EV, AC, PV)'
                elif device_count == 2:
                    icon_color = 'red'
                    icon_name = 'plus'
                    tooltip_device = 'Two Devices'
                elif device_count == 1:
                    # Corrected indentation for inner if/elifs
                    if house['has_ev'] and show_ev:
                        icon_color = "blue"
                        icon_name = "plug"
                        tooltip_device = "EV Charger Only"
                    elif house['has_pv'] and show_pv:
                        icon_color = "green"
                        icon_name = "sun"
                        tooltip_device = "Solar Panel Only"
                    elif house['has_ac'] and show_ac:
                        icon_color = "orange"
                        icon_name = "snowflake"
                        tooltip_device = "AC Unit Only"
                    else:
                        # Single device, but its filter is off - Corrected indentation
                        icon_color = 'lightgray' 
                        icon_name = 'home'
                        tooltip_device = 'Single Device (Filtered Out)'
                else: # device_count == 0 - Corrected indentation
                    icon_color = 'gray'
                    icon_name = 'question'
                    tooltip_device = 'No Devices Detected'
                
                popup_content = f"""
                <div style="min-width: 180px;">
                    <h4>Home {house['id']}</h4>
                    <b>Detected Devices:</b><br>
                    {'<span style="color: blue;">✓ EV Charger</span><br>' if house['has_ev'] else ''}
                    {'<span style="color: orange;">✓ AC Unit</span><br>' if house['has_ac'] else ''}
                    {'<span style="color: green;">✓ Solar Panels</span><br>' if house['has_pv'] else ''}
                    {'✓ EV Charger<br>' if house['has_ev'] else ''}
                    {'✓ AC Unit<br>' if house['has_ac'] else ''}
                    {'✓ Solar Panels<br>' if house['has_pv'] else ''}
                    {f"<b>State:</b> {house['state']}<br>" if 'state' in house and pd.notna(house['state']) else ''}
                    {f"<b>Coords:</b> ({house['lat']:.4f}, {house['lon']:.4f})"}
                </div>
                """
                
                # Add marker
                folium.Marker(
                    location=[house['lat'], house['lon']],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa'),
                    tooltip=f"Home {house['id']} ({tooltip_device})"
                ).add_to(marker_cluster)
        else:
            st.info(f"No households match the selected filters for {selected_state_name}.")

    # --- Display Legend --- 
    st.markdown("#### Map Legend")
    legend_cols = st.columns([1, 1, 1, 1, 1]) 
    with legend_cols[0]:
        if show_ev: st.markdown('<i class="fa fa-plug" style="color: blue;"></i> EV Only', unsafe_allow_html=True)
    with legend_cols[1]:
        if show_ac: st.markdown('<i class="fa fa-snowflake" style="color: orange;"></i> AC Only', unsafe_allow_html=True)
    with legend_cols[2]:
        if show_pv: st.markdown('<i class="fa fa-sun" style="color: green;"></i> PV Only', unsafe_allow_html=True)
    with legend_cols[3]:
        st.markdown('<i class="fa fa-plus" style="color: red;"></i> Two Devices', unsafe_allow_html=True)
    with legend_cols[4]:
        st.markdown('<i class="fa fa-star" style="color: purple;"></i> All Three', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Display the Map --- 
    st.subheader("Map View")
    if not selected_state_code:
        st.caption("Select a state from the dropdown above to load household markers.")
    # Captions for filtered states moved inside the if block below

    folium.LayerControl().add_to(m)
    folium_static(m, width=1000, height=600)

    # --- Statistics Display --- 
    st.markdown("---") 
    if selected_state_code and not all_household_data.empty:
        state_data = all_household_data[all_household_data['state'].astype(str).str.upper() == selected_state_code]
        if not state_data.empty:
            st.subheader(f"{selected_state_name} Statistics")
            total_homes = len(state_data)
            ev_homes = state_data['has_ev'].sum()
            ac_homes = state_data['has_ac'].sum()
            pv_homes = state_data['has_pv'].sum()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Homes", f"{total_homes:,}")
            with col2:
                ev_pct = (ev_homes / total_homes) if total_homes > 0 else 0
                st.metric("EV Chargers", f"{ev_homes:,}", f"{ev_pct:.1%}")
            with col3:
                ac_pct = (ac_homes / total_homes) if total_homes > 0 else 0
                st.metric("AC Units", f"{ac_homes:,}", f"{ac_pct:.1%}")
            with col4:
                pv_pct = (pv_homes / total_homes) if total_homes > 0 else 0
                st.metric("Solar Panels", f"{pv_homes:,}", f"{pv_pct:.1%}")

            # Display data table - Corrected indentation
            if not filtered_households_df.empty:
                st.subheader("Filtered Homes Data")
                columns_to_show = ['id', 'lat', 'lon', 'has_ev', 'has_ac', 'has_pv', 'state'] 
                df_display = filtered_households_df[[col for col in columns_to_show if col in filtered_households_df.columns]].copy()
                
                column_map = {
                    'id': 'Home ID',
                    'lat': 'Latitude',
                    'lon': 'Longitude',
                    'has_ev': 'EV Charger',
                    'has_ac': 'AC Unit',
                    'has_pv': 'Solar Panel',
                    'state': 'State' 
                }
                # Corrected indentation
                df_display = df_display.rename(columns=column_map)
                for col in ['EV Charger', 'AC Unit', 'Solar Panel']:
                    if col in df_display.columns:
                        df_display[col] = df_display[col].map({True: 'Yes', False: 'No'})

                if 'Latitude' in df_display.columns: df_display['Latitude'] = df_display['Latitude'].map('{:.4f}'.format)
                if 'Longitude' in df_display.columns: df_display['Longitude'] = df_display['Longitude'].map('{:.4f}'.format)

                st.dataframe(df_display, use_container_width=True, height=300)
                # Add caption specific to filtered data being shown
                st.caption(f"Showing {len(filtered_households_df)} households matching filters for {selected_state_name}.")
            else:
                # Add caption when filters result in no data for the selected state
                st.caption(f"No households match the selected device filters for {selected_state_name}.")

        else:
            st.info(f"No household data found for {selected_state_name} in the provided CSV.")
    else:
        # Keep this for when no state is selected at all
        if not selected_state_code:
             st.subheader("Portfolio Overview")
             st.info("Select a state from the dropdown above to view detailed statistics and household data.")

    st.markdown("---")
    # Updated footer text slightly
    st.markdown(f"""
    <div style="text-align:center; color:{primary_purple}; padding: 10px; border-radius: 5px;">
        Use the dropdown to select a state and explore homes with detected devices.
        Household location data is illustrative.
    </div>
    """, unsafe_allow_html=True)
