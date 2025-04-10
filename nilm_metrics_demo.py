import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import random
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
import json
import geopandas as gpd
from folium.plugins import MarkerCluster
import plotly.subplots as sp
from PIL import Image
import datetime
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

# Add page selection at the top (now with three options)
page = st.sidebar.radio("Select Page", ["Sample Output", "Performance Metrics", "Interactive Map"], index=1)

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
    # Sample Output page code goes here
    st.title("Energy Disaggregation Model: Output Structure Example")

    # Define function to convert date string to Unix timestamp string (start of day/month)
    def to_unix_timestamp_string(date_str):
        """Converts various date string formats to a Unix timestamp string (start of day/month, UTC)."""
        if pd.isna(date_str):
            return None
        try:
            # Assuming date_str is in a format pandas can parse (e.g., YYYY-MM-DD or similar)
            dt = pd.to_datetime(date_str)
            # Convert to UTC first to ensure consistency
            # Use the string 'UTC' for localization
            dt_utc = dt.tz_localize(None).tz_localize('UTC') 
            # For reference month or window start/stop, get the timestamp of the first day of the month UTC
            dt_month_start_utc = dt_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            return str(int(dt_month_start_utc.timestamp()))
        except Exception as e:
            # Add warning to help debug problematic date values
            st.warning(f"Could not convert date '{date_str}' to timestamp: {e}") 
            return None # Handle parsing errors gracefully

    try:
        # --- Load Data --- 
        # IMPORTANT: Adjust file paths as needed
        metadata_csv_path = 'dissag_output_v2.csv' 
        consumption_csv_path = 'disagg_sample.csv'
        
        try:
            df_metadata = pd.read_csv(metadata_csv_path)
            # st.write("Metadata CSV loaded:", df_metadata.head()) # Debug
        except FileNotFoundError:
            st.error(f"Error: Metadata file '{metadata_csv_path}' not found.")
            st.stop()
        except Exception as e:
            st.error(f"Error reading metadata CSV '{metadata_csv_path}': {e}")
            st.stop()

        try:
            df_consumption = pd.read_csv(consumption_csv_path)
            # st.write("Consumption CSV loaded:", df_consumption.head()) # Debug
        except FileNotFoundError:
            st.error(f"Error: Consumption file '{consumption_csv_path}' not found.")
            st.stop()
        except Exception as e:
            st.error(f"Error reading consumption CSV '{consumption_csv_path}': {e}")
            st.stop()

        st.markdown("""
        This page shows the restructured output format using data merged from two sources.
        Dates are represented as Unix timestamp strings (UTC, start of day/month).
        """)

        # --- Prepare Metadata (df_metadata) --- 
        meta_rename_map = {
            'dataid': 'meterid', # EXPECTED key in new file
            'start_date_disagg_window': 'window_start_str', 
            'end_date_disagg_window': 'window_stop_str',
            'start_date_output_window': 'reference_month_str'
        }
        actual_meta_rename = {k: v for k, v in meta_rename_map.items() if k in df_metadata.columns}
        df_metadata = df_metadata.rename(columns=actual_meta_rename)
        
        required_meta_cols = ['meterid', 'window_start_str', 'window_stop_str', 'reference_month_str']
        missing_meta = [col for col in required_meta_cols if col not in df_metadata.columns]
        if missing_meta:
            st.error(f"Error: Metadata CSV '{metadata_csv_path}' is missing required columns (after potential rename): {missing_meta}. Expected original names: {list(meta_rename_map.keys())}")
            st.stop()
            
        # Convert dates to timestamps
        df_metadata['window_start_ts'] = df_metadata['window_start_str'].apply(to_unix_timestamp_string)
        df_metadata['window_stop_ts'] = df_metadata['window_stop_str'].apply(to_unix_timestamp_string)
        df_metadata['reference_month_ts'] = df_metadata['reference_month_str'].apply(to_unix_timestamp_string)
        
        # Keep only necessary metadata columns
        df_metadata = df_metadata[['meterid', 'window_start_ts', 'window_stop_ts', 'reference_month_ts']]
        if df_metadata.isnull().values.any():
             st.warning("Warning: Some dates in metadata could not be converted to timestamps.")
        # st.write("Prepared Metadata:", df_metadata.head()) # Debug
             
        # --- Prepare Consumption Data (df_consumption) ---
        # EXPECTED key in old file (maps to dataid/meterid in new file)
        consum_id_col = 'dataid' 
        if consum_id_col not in df_consumption.columns:
             st.error(f"Error: Consumption CSV '{consumption_csv_path}' is missing the identifier column '{consum_id_col}'. Cannot merge data.")
             st.stop()
        df_consumption = df_consumption.rename(columns={consum_id_col: 'meterid'})
        
        # Identify grid column
        grid_col = 'grid (kWh)'
        if grid_col not in df_consumption.columns:
             grid_col = 'grid'
             if grid_col not in df_consumption.columns:
                 st.warning(f"Consumption CSV '{consumption_csv_path}' missing grid column ('grid (kWh)' or 'grid'). Grid values will be omitted.")
                 grid_col = None
                 
        # Identify appliance columns
        appliance_map = {
            'ev charging (kWh)': 'a', 
            'solar production (kWh)': 'b', 
            'air conditioning (kWh)': 'c', 
            'water heater (kWh)': 'd'  
        }
        appliance_cols = [col for col in appliance_map.keys() if col in df_consumption.columns]
        if not appliance_cols:
             st.error(f"Error: Consumption CSV '{consumption_csv_path}' is missing expected appliance kWh columns (e.g., 'ev charging (kWh)').")
             st.stop()
             
        # Keep only necessary consumption columns
        cols_to_keep = ['meterid'] + appliance_cols
        if grid_col: 
            cols_to_keep.append(grid_col)
        df_consumption = df_consumption[cols_to_keep]
        # st.write("Prepared Consumption Data:", df_consumption.head()) # Debug

        # --- Merge Data --- 
        try:
            df_merged = pd.merge(df_metadata, df_consumption, on='meterid', how='inner')
        except Exception as e:
             st.error(f"Error merging metadata and consumption data on 'meterid': {e}")
             st.stop()
             
        if df_merged.empty:
             st.warning("Warning: No matching 'meterid's found between the two CSV files. Cannot display merged data.")
             st.stop()
        # st.write("Merged Data:", df_merged.head()) # Debug
             
        # --- Create Table 1: Meter Information --- 
        st.subheader("Table 1: Meter Information")
        meter_info_cols = ['meterid', 'window_start_ts', 'window_stop_ts']
        try:
            # Select columns *before* drop_duplicates
            meter_info_df = df_merged[meter_info_cols].drop_duplicates(subset=['meterid']).reset_index(drop=True)
            meter_info_df['interval'] = "15 min" 
            # Rename timestamp columns for display
            meter_info_df = meter_info_df.rename(columns={'window_start_ts':'window_start', 'window_stop_ts':'window_stop'})
            st.dataframe(meter_info_df, use_container_width=True)
        except KeyError as e:
            st.error(f"Error creating Meter Info table (KeyError): {e}. This might indicate missing columns post-merge.")
        except Exception as e:
            st.error(f"An unexpected error occurred creating Meter Info table: {e}")
            st.exception(e)
            
        # --- Create Table 2: Appliance Breakdown --- 
        # (Melt and prepare data FIRST, then calculate plots/metrics)
        id_vars_melt = ['meterid', 'reference_month_ts']
        if grid_col and grid_col in df_merged.columns: 
             id_vars_melt.append(grid_col)
             
        value_vars_melt = [col for col in appliance_cols if col in df_merged.columns]
        if not value_vars_melt:
             st.warning("Warning: No appliance columns found in the merged data. Cannot create breakdown table or plots.")
             # Optionally stop, or allow proceeding without breakdown/plots
             appliance_breakdown_df = pd.DataFrame() # Create empty df
             # Set flags or default values for plot data if needed
        else:
            try:
                # Melt the merged dataframe
                appliance_breakdown_df = pd.melt(
                    df_merged, 
                    id_vars=id_vars_melt,
                    value_vars=value_vars_melt, 
                    var_name='original_appliance_col', 
                    value_name='Consumption (kWh)'
                )
                
                # Convert consumption to numeric, handle errors/NaNs
                appliance_breakdown_df['Consumption (kWh)'] = pd.to_numeric(appliance_breakdown_df['Consumption (kWh)'], errors='coerce').fillna(0)
            
            except Exception as e:
                 st.error(f"Error during melting or initial processing for Table 2: {e}")
                 appliance_breakdown_df = pd.DataFrame() # Ensure it exists but is empty on error

        # --- Calculate Data for Plots and Metrics --- (Using df_merged and appliance_breakdown_df)
        total_unique_meters = df_merged['meterid'].nunique()
        appliance_presence = {}
        energy_totals = {}
        avg_grid = 0
        avg_pv_prod = 0
        pct_identified = 0
        total_identified = 0
        total_grid = 0

        if not appliance_breakdown_df.empty and total_unique_meters > 0:
            # Create appliance type code column early for calculations
            appliance_code_map_calc = {k: v for k, v in appliance_map.items() if k in value_vars_melt}
            appliance_breakdown_df['appliance_type'] = appliance_breakdown_df['original_appliance_col'].map(appliance_code_map_calc)
            
            # Filter for positive consumption for calculations
            positive_consumption_df = appliance_breakdown_df[appliance_breakdown_df['Consumption (kWh)'] > 0]
            
            # Calculate Presence (% of unique meters having the appliance with >0 kWh)
            presence_counts = positive_consumption_df.groupby('appliance_type')['meterid'].nunique()
            appliance_presence = { # Map codes back to full names for plot labels
                'EV Charging': (presence_counts.get('a', 0) / total_unique_meters) * 100,
                'Solar PV': (presence_counts.get('b', 0) / total_unique_meters) * 100,
                'Air Conditioning': (presence_counts.get('c', 0) / total_unique_meters) * 100,
                'Water Heater': (presence_counts.get('d', 0) / total_unique_meters) * 100,
            }

            # Calculate Energy Totals (Sum of positive kWh for EV, AC, WH)
            energy_sum_by_type = positive_consumption_df.groupby('appliance_type')['Consumption (kWh)'].sum()
            ev_total = energy_sum_by_type.get('a', 0)
            ac_total = energy_sum_by_type.get('c', 0)
            wh_total = energy_sum_by_type.get('d', 0)
            total_identified = ev_total + ac_total + wh_total
            
            # Calculate Total Grid (Sum unique grid values per meter/month or just sum column if appropriate)
            # Assuming grid value is per meter/month in the original merged data
            if grid_col and grid_col in df_merged.columns:
                # Sum unique grid values per meter/month combination if needed
                # For simplicity, summing the column in df_merged (might overcount if multiple rows per meter/month before melt)
                # A safer approach would be df_merged[[unique_key_cols, grid_col]].drop_duplicates()[grid_col].sum()
                # Let's assume for now df_merged has one row per meter/month before melt for grid value
                total_grid = pd.to_numeric(df_merged[grid_col], errors='coerce').fillna(0).sum()
                avg_grid = pd.to_numeric(df_merged[grid_col], errors='coerce').mean() 
            else: 
                total_grid = 0 # Grid column missing
                avg_grid = 0

            other_consumption = max(0, total_grid - total_identified)
            energy_totals = {
                'EV Charging': ev_total,
                'Air Conditioning': ac_total,
                'Water Heater': wh_total,
                'Other Consumption': other_consumption
            }
            
            # Calculate Avg PV Production (Avg positive kWh for PV)
            pv_prod = positive_consumption_df[positive_consumption_df['appliance_type'] == 'b']['Consumption (kWh)']
            avg_pv_prod = pv_prod.mean() if not pv_prod.empty else 0
            
            # Calculate % Consumption Identified
            pct_identified = (total_identified / total_grid * 100) if total_grid > 0 else 0
        else:
             st.info("No appliance data found to calculate plots and metrics.")

        # --- Display Plots and Metrics --- (Re-introducing)
        st.markdown("### Visualization and Key Metrics")
        plot_col1, plot_col2 = st.columns(2)

        with plot_col1:
            st.subheader("Appliance Presence")
            if appliance_presence:
                fig1 = px.bar(
                    x=list(appliance_presence.keys()),
                    y=list(appliance_presence.values()),
                    labels={'x': 'Appliance Type', 'y': 'Percentage of Homes (%)'},
                    color=list(appliance_presence.keys()),
                    color_discrete_map={
                            'EV Charging': primary_purple, 'Air Conditioning': green,
                            'Solar PV': cream, 'Water Heater': salmon
                    },
                    text=[f"{val:.1f}%" for val in appliance_presence.values()]
                )
                fig1.update_layout(
                    showlegend=False, yaxis_range=[0, 100],
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor=white, plot_bgcolor=white, font=dict(color=dark_purple)
                )
                fig1.update_traces(textposition='outside', textfont=dict(color=dark_purple))
                fig1.update_xaxes(showgrid=False, tickfont=dict(color=dark_purple))
                fig1.update_yaxes(showgrid=True, gridcolor=light_purple, tickfont=dict(color=dark_purple))
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.caption("Presence data not available.")

        with plot_col2:
            st.subheader("Total Energy Distribution")
            if energy_totals and sum(energy_totals.values()) > 0:
                fig2 = px.pie(
                    values=list(energy_totals.values()),
                    names=list(energy_totals.keys()),
                    color=list(energy_totals.keys()),
                    color_discrete_map={
                            'EV Charging': primary_purple, 'Air Conditioning': green,
                            'Water Heater': salmon, 'Other Consumption': light_purple
                    },
                    hole=0.4
                )
                fig2.update_layout(
                    legend_title="Energy Type", margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor=white, plot_bgcolor=white, font=dict(color=dark_purple),
                    legend=dict(font=dict(color=dark_purple))
                )
                fig2.update_traces(
                    textinfo='percent+label', hovertemplate='%{label}<br>%{value:.1f} kWh<br>%{percent}',
                    textfont=dict(color=dark_gray) # Darker text for contrast
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                 st.caption("Energy distribution data not available.")

        # Key Metrics Section
        st.subheader("Key Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric(label="Total Homes", value=f"{total_unique_meters:,}")
        with metric_col2:
            st.metric(label="Avg. Grid Consumption", value=f"{avg_grid:.1f} kWh")
        with metric_col3:
            st.metric(label="Avg. Solar Production (PV)", value=f"{avg_pv_prod:.1f} kWh", help="Average production for homes with detected PV (>0 kWh)")
        with metric_col4:
            st.metric(label="Consumption Identified", value=f"{pct_identified:.1f}%", help="% of total grid kWh attributed to detected EV, AC, and WH")

        # --- Prepare and Display Table 2 --- 
        st.subheader("Table 2: Appliance Breakdown")
        if not appliance_breakdown_df.empty:
            try:
                # Set non-positive consumption to -1 (if not already done, depends on calculation flow)
                appliance_breakdown_df.loc[appliance_breakdown_df['Consumption (kWh)'] <= 0, 'Consumption (kWh)'] = -1
                
                # Ensure appliance_type column exists (calculated above)
                if 'appliance_type' not in appliance_breakdown_df.columns:
                     # Recalculate if it was lost somehow
                     appliance_code_map_disp = {k: v for k, v in appliance_map.items() if k in value_vars_melt}
                     appliance_breakdown_df['appliance_type'] = appliance_breakdown_df['original_appliance_col'].map(appliance_code_map_disp)
                
                # Select, rename, and reorder final columns for display
                final_display_cols = ['meterid', 'appliance_type']
                final_display_rename_map = {'reference_month_ts': 'reference_month'}
                
                # Check if grid column exists in the breakdown df before adding
                # Use the original name before potential renaming in map
                grid_col_in_breakdown = grid_col if grid_col and grid_col in appliance_breakdown_df.columns else None
                if grid_col_in_breakdown:
                    final_display_cols.append(grid_col_in_breakdown) 
                    final_display_rename_map[grid_col_in_breakdown] = 'grid (kWh)' # RENAME to include (kWh)
                
                final_display_cols.extend(['Consumption (kWh)', 'reference_month_ts'])
                
                # Select only the necessary columns that exist
                final_display_cols_exist = [c for c in final_display_cols if c in appliance_breakdown_df.columns]
                display_df = appliance_breakdown_df[final_display_cols_exist].copy() # Work on a copy
                
                # Rename the selected columns
                display_df = display_df.rename(columns=final_display_rename_map)
                
                # Define final order based on RENAMED columns
                final_display_order = ['meterid', 'appliance_type']
                if 'grid (kWh)' in display_df.columns: # Check using the new name
                    final_display_order.append('grid (kWh)')
                final_display_order.extend(['Consumption (kWh)', 'reference_month']) 
                
                # Ensure all columns in final order exist before reordering
                final_display_order_exist = [c for c in final_display_order if c in display_df.columns]
                display_df = display_df[final_display_order_exist]
                
                st.dataframe(display_df, use_container_width=True)
                
                # Add legend for appliance codes
                st.caption("Appliance Type Codes: a = EV Charging, b = Solar PV, c = Air Conditioning, d = Water Heater")

            except Exception as e:
                st.error(f"An error occurred preparing Table 2 for display: {e}")
                st.exception(e)
        else:
             st.info("No data available for Appliance Breakdown Table.")

    except Exception as e:
        st.error(f"An unexpected error occurred on the Sample Output page: {e}")
        st.exception(e)

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
        best_overall_display = trend_df.loc[trend_df['avg_score'].idxmax()]['Model']
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
    
    # --- Refactored Data Generation ---
    
    # Cache state information (coordinates, name, zoom)
    @st.cache_data
    def get_states_info():
        # US states with coordinates (approximate centers) - all 50 states plus DC
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
        # Generate stats for each state (needed for filtering/tooltips later if desired)
        for state_code in states_data:
            state = states_data[state_code]
            state['total_homes'] = random.randint(150, 500) # Keep for potential stats display
            state['ev_homes'] = random.randint(30, int(state['total_homes'] * 0.3))
            state['ac_homes'] = random.randint(int(state['total_homes'] * 0.5), int(state['total_homes'] * 0.9))
            state['pv_homes'] = random.randint(20, int(state['total_homes'] * 0.25))
        return states_data
        
    # Cache household generation per state
    @st.cache_data
    def generate_households_for_state(state_code, states_info):
        """Generate mock household data within a specific state"""
        if state_code not in states_info:
            return []
            
        state = states_info[state_code]
        households = []
        count = state['total_homes'] # Use the pre-calculated total homes

        # Define the spread of points (in degrees)
        lat_spread = 1.5
        lon_spread = 1.5

        for i in range(count):
            # Randomly place homes around the state center
            lat = state['lat'] + (random.random() - 0.5) * lat_spread
            lon = state['lon'] + (random.random() - 0.5) * lon_spread

            # Assign devices randomly but weighted by state percentages
            # Use percentages directly from states_info for consistency
            ev_prob = state['ev_homes'] / state['total_homes']
            ac_prob = state['ac_homes'] / state['total_homes']
            pv_prob = state['pv_homes'] / state['total_homes']

            has_ev = random.random() < ev_prob
            has_ac = random.random() < ac_prob
            has_pv = random.random() < pv_prob

            # Ensure at least one device is present if filters require it (simplification)
            # This logic might need refinement depending on exact requirements
            # For now, ensure *something* is true if needed by simple assignment
            if not (has_ev or has_ac or has_pv):
                 # If all are false, randomly set one based on probability as a fallback
                 rand_choice = random.random()
                 if rand_choice < ev_prob: has_ev = True
                 elif rand_choice < ev_prob + ac_prob: has_ac = True
                 else: has_pv = True


            household = {
                'id': f"{state_code}-{i+1}",
                'lat': lat,
                'lon': lon,
                'has_ev': has_ev,
                'has_ac': has_ac,
                'has_pv': has_pv,
                'energy_consumption': random.randint(20, 100),
                'state': state_code
            }
            households.append(household)
        
        return households
        
    # --- New Function: Generate Historical Data ---
    @st.cache_data
    def generate_historical_data_for_state(state_code, states_info, num_years=5):
        """Generates simulated historical appliance adoption percentages for a state."""
        if state_code not in states_info:
            return pd.DataFrame()

        state = states_info[state_code]
        current_year = datetime.datetime.now().year
        years = list(range(current_year - num_years + 1, current_year + 1))
        
        # Get current percentages
        current_ev_pct = (state['ev_homes'] / state['total_homes']) * 100
        current_ac_pct = (state['ac_homes'] / state['total_homes']) * 100
        current_pv_pct = (state['pv_homes'] / state['total_homes']) * 100

        data = []
        
        # Simulate historical data with upward trend (especially EV, PV) and noise
        for year in years:
            year_factor = (year - (current_year - num_years)) / num_years # 0 to 1 scale over years

            # Simulate EV: lower start, strong growth + noise
            ev_start_factor = 0.2 + random.uniform(-0.1, 0.1) # Start around 20% of current value
            ev_pct = max(1, current_ev_pct * (ev_start_factor + (1 - ev_start_factor) * year_factor**1.5) + random.uniform(-2, 2))
            data.append({'Year': year, 'Device': 'EV Chargers', 'Percentage': ev_pct})

            # Simulate AC: higher start, slower growth + noise
            ac_start_factor = 0.8 + random.uniform(-0.05, 0.05) # Start around 80% of current value
            ac_pct = min(95, max(10, current_ac_pct * (ac_start_factor + (1 - ac_start_factor) * year_factor**0.8) + random.uniform(-3, 3)))
            data.append({'Year': year, 'Device': 'AC Units', 'Percentage': ac_pct})

            # Simulate PV: lower start, moderate growth + noise
            pv_start_factor = 0.3 + random.uniform(-0.1, 0.1) # Start around 30% of current value
            pv_pct = max(1, current_pv_pct * (pv_start_factor + (1 - pv_start_factor) * year_factor**1.2) + random.uniform(-1.5, 1.5))
            data.append({'Year': year, 'Device': 'Solar Panels', 'Percentage': pv_pct})

        hist_df = pd.DataFrame(data)
        # Ensure percentages are within reasonable bounds (0-100)
        hist_df['Percentage'] = hist_df['Percentage'].clip(0, 100)
        return hist_df

    # Load GeoJSON data for US states
    @st.cache_data
    def load_us_geojson():
        try:
            response = requests.get("https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json")
            response.raise_for_status() # Raise an exception for bad status codes
            us_states = response.json()
            return us_states
        except requests.exceptions.RequestException as e:
            st.error(f"Error loading US state boundaries: {e}")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred while processing GeoJSON: {e}")
            return None

    # --- Map Generation and Display Logic ---
    # Create filter controls in sidebar
    st.sidebar.markdown("### Map Filters")
    show_ev = st.sidebar.checkbox("Show EV Chargers", value=True)
    show_ac = st.sidebar.checkbox("Show AC Units", value=True)
    show_pv = st.sidebar.checkbox("Show Solar Panels", value=True)
    
    st.sidebar.markdown("---") # Separator
    
    # Add custom styling for the refresh button (optional, kept from previous code)
    st.sidebar.markdown("""
    <style>
    div[data-testid="stButton"] button {
        background-color: #515D9A !important;
        color: black !important; /* Changed to black for better contrast on light button */
        font-weight: bold !important;
        border: 1px solid darkgray !important;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #B8BCF3 !important;
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add refresh button to sidebar (clears state selection)
    if st.sidebar.button("↻ Reset Map View", use_container_width=True):
        # Clear state selection from session state if it exists
        if 'selected_state_map' in st.session_state:
            del st.session_state['selected_state_map']
        st.rerun()
    
    # Load data
    states_info = get_states_info()
    us_geojson = load_us_geojson()
    
    # --- State Selection ---
    state_names = {code: info['name'] for code, info in states_info.items()}
    state_codes_sorted = sorted(state_names.keys(), key=lambda k: state_names[k])
    state_options = ["Select a State..."] + [state_names[code] for code in state_codes_sorted]
    
    # Use session state to remember the selection
    if 'selected_state_map' not in st.session_state:
        st.session_state['selected_state_map'] = state_options[0] # Default to "Select a State..."
        
    selected_state_name = st.selectbox(
        "Select a State to View Households:",
        options=state_options,
        key='selected_state_map', # Persist selection across reruns
        index=state_options.index(st.session_state['selected_state_map']) # Set current index
    )

    # Find the state code corresponding to the selected name
    selected_state_code = None
    if selected_state_name != "Select a State...":
        for code, name in state_names.items():
            if name == selected_state_name:
                selected_state_code = code
                break

    # --- Create Map ---
    map_location = [39.8283, -98.5795] # Default: Center of US
    map_zoom = 4
    
    # If a state is selected, update map center and zoom
    if selected_state_code:
        state_info = states_info[selected_state_code]
        map_location = [state_info['lat'], state_info['lon']]
        map_zoom = state_info['zoom']
        
    # Initialize map
    m = folium.Map(
        location=map_location, 
        zoom_start=map_zoom,
        tiles="CartoDB positron" # Use a light base map
    )

    # Add satellite view layer
    folium.TileLayer('Esri_WorldImagery', name='Satellite View', attr='Esri').add_to(m)
    
    # Add state boundaries (GeoJSON) - always visible
    if us_geojson:
        style_function = lambda x: {
            'fillColor': primary_purple,
            'color': 'white',
            'weight': 1,
            'fillOpacity': 0.3 if selected_state_code else 0.5 # Make less opaque if state selected
        }
        highlight_function = lambda x: {
                'fillColor': light_purple,
                'color': 'white',
                'weight': 3,
                'fillOpacity': 0.7
        } if not selected_state_code else style_function(x) # No highlight if state selected

        folium.GeoJson(
            us_geojson,
            name='State Boundaries',
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['State:'], labels=True, sticky=False),
            # Remove popup click functionality as selection is via dropdown
            # popup=folium.GeoJsonPopup(...) 
        ).add_to(m)

    # --- Load and Add Household Markers (Only if a state is selected) ---
    filtered_households = []
    if selected_state_code:
        with st.spinner(f"Loading household data for {selected_state_name}..."):
            # Generate or retrieve cached households for the selected state
            state_households = generate_households_for_state(selected_state_code, states_info)
            
            # Apply device filters from sidebar
            filtered_households = [
                h for h in state_households if 
                ((show_ev and h['has_ev']) or 
                 (show_ac and h['has_ac']) or 
                 (show_pv and h['has_pv']))
            ]

        if filtered_households:
            # Create marker cluster for households
            marker_cluster = MarkerCluster(name=f'{selected_state_name} Homes').add_to(m)
            
            # Add markers for each household
            for house in filtered_households:
                # Determine marker icon based on devices (priority: EV > PV > AC)
                if house['has_ev'] and show_ev:
                    icon_color = "blue"
                    icon_name = "plug"
                    tooltip_device = "EV Charger"
                elif house['has_pv'] and show_pv:
                    icon_color = "green"
                    icon_name = "sun"
                    tooltip_device = "Solar Panel"
                elif house['has_ac'] and show_ac:
                     icon_color = "orange"
                     icon_name = "snowflake" # Changed AC icon
                     tooltip_device = "AC Unit"
                else:
                    # This case should ideally not happen due to filtering, but as fallback:
                    icon_color = "gray" 
                    icon_name = "home"
                    tooltip_device = "Other"


                # Create popup content
                popup_content = f"""
                <div style="min-width: 200px;">
                    <h4>Home {house['id']}</h4>
                    <b>Devices:</b><br>
                    {'✓ EV Charger<br>' if house['has_ev'] else ''}
                    {'✓ AC Unit<br>' if house['has_ac'] else ''}
                    {'✓ Solar Panels<br>' if house['has_pv'] else ''}
                    <b>Daily Energy:</b> {house['energy_consumption']} kWh
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

    # Add legend (always visible)
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; right: 50px; 
                background-color: rgba(255, 255, 255, 0.8); /* Slightly transparent background */
                padding: 10px; border-radius: 5px;
                border: 1px solid #ccc; z-index: 9999; font-size: 12px;">
        <h4 style="margin-top: 0; margin-bottom: 5px; color: #515D9A; text-align: center;">Legend</h4>
        {'<div style="display: flex; align-items: center; margin-bottom: 3px;"><i class="fa fa-plug" style="color: blue; margin-right: 8px; font-size: 14px;"></i> EV Charger</div>' if show_ev else ''}
        {'<div style="display: flex; align-items: center; margin-bottom: 3px;"><i class="fa fa-snowflake" style="color: orange; margin-right: 8px; font-size: 14px;"></i> AC Unit</div>' if show_ac else ''}
        {'<div style="display: flex; align-items: center;"><i class="fa fa-sun" style="color: green; margin-right: 8px; font-size: 14px;"></i> Solar Panel</div>' if show_pv else ''}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add Layer Control to toggle layers (Satellite, States, Homes cluster if present)
    folium.LayerControl().add_to(m)
    
    # Display the map
    st.subheader("Map View")
    if not selected_state_code:
         st.caption("Select a state from the dropdown above to load household markers.")
    elif not filtered_households:
         st.caption(f"Showing map for {selected_state_name}. No households match the current filters.")
    else:
         st.caption(f"Showing {len(filtered_households)} households for {selected_state_name}. Use sidebar filters to refine.")
         
    folium_static(m, width=1000, height=600) # Consider adjusting width/height if needed
    
    # --- Statistics Display ---
    st.markdown("---") # Separator before stats

    # Display statistics ONLY if a state is selected
    if selected_state_code:
        state = states_info[selected_state_code]
        st.subheader(f"{state['name']} Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Homes", f"{state['total_homes']:,}")
        with col2:
            ev_pct = state['ev_homes']/state['total_homes']
            st.metric("EV Chargers", f"{state['ev_homes']:,}", f"{ev_pct:.1%}")
        with col3:
            ac_pct = state['ac_homes']/state['total_homes']
            st.metric("AC Units", f"{state['ac_homes']:,}", f"{ac_pct:.1%}")
        with col4:
            pv_pct = state['pv_homes']/state['total_homes']
            st.metric("Solar Panels", f"{state['pv_homes']:,}", f"{pv_pct:.1%}")
        
        # Display data table for the filtered households shown on map
        if filtered_households:
            st.subheader("Filtered Homes Data")
            columns_to_show = ['id', 'energy_consumption', 'has_ev', 'has_ac', 'has_pv']
            df = pd.DataFrame(filtered_households)[columns_to_show]
            
            # Rename columns for display
            column_map = {
                'id': 'Home ID',
                'energy_consumption': 'Energy (kWh)',
                'has_ev': 'EV Charger',
                'has_ac': 'AC Unit',
                'has_pv': 'Solar Panel'
            }
            df = df.rename(columns=column_map)
            # Convert booleans to Yes/No for better readability
            df['EV Charger'] = df['EV Charger'].map({True: 'Yes', False: 'No'})
            df['AC Unit'] = df['AC Unit'].map({True: 'Yes', False: 'No'})
            df['Solar Panel'] = df['Solar Panel'].map({True: 'Yes', False: 'No'})

            st.dataframe(df, use_container_width=True, height=300) # Added height limit
        else:
             st.write("No data to display for the current filter selection.")

        # --- Add Historical Trend Section ---
        st.markdown("---") # Separator
        st.subheader(f"Historical Appliance Trends for {state['name']}")
        st.caption("Simulated adoption rates over the last 5 years.")

        # Generate historical data
        historical_df = generate_historical_data_for_state(selected_state_code, states_info, num_years=5)

        if not historical_df.empty:
            # Create line chart
            fig_hist = px.line(
                historical_df,
                x='Year',
                y='Percentage',
                color='Device',
                markers=True,
                labels={'Percentage': 'Adoption Rate (%)', 'Device': 'Appliance Type'},
                color_discrete_map={
                        'EV Chargers': primary_purple, # Use existing color scheme
                        'AC Units': green,
                        'Solar Panels': cream # Or perhaps another distinct color like salmon? Let's use cream for now.
                },
                template='plotly_white' # Use a clean template
            )

            fig_hist.update_layout(
                yaxis_range=[0, 100],
                xaxis=dict(tickmode='linear'), # Ensure all years are shown
                legend_title="Appliance Type",
                paper_bgcolor=white,
                plot_bgcolor=white,
                font=dict(color=dark_purple)
            )
            
            fig_hist.update_traces(marker=dict(size=8))

            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("Could not generate historical data.")
        # --- End Historical Trend Section ---


    else:
        # Optionally show overall stats or a message when no state is selected
        st.subheader("Portfolio Overview")
        st.info("Select a state from the dropdown above to view detailed statistics and household data.")
        # You could potentially calculate and display aggregate stats here if needed
        # total_homes = sum(info['total_homes'] for info in states_info.values())
        # ... etc ...

    # --- End Map Generation Logic ---

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; color:{primary_purple}; padding: 10px; border-radius: 5px;">
        Use the dropdown to select a state and explore homes with EV chargers, AC units, and solar panels.
        Map data is randomly generated for demonstration purposes.
    </div>
    """, unsafe_allow_html=True)
