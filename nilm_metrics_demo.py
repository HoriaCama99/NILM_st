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

if page == "Sample Output":
    # Sample Output page code goes here
    st.title("Energy Disaggregation Model: Sample Output")
    
    # Try to load the sample data CSV
    try:
        df = pd.read_csv('disagg_sample.csv')
        
        st.markdown("""
        This dashboard presents a sample output from our energy disaggregation model, which analyzes household 
        energy consumption data and identifies specific appliance usage patterns.

        ### Key Assumptions:
        - Sample represents output for multiple homes with diverse energy profiles
        - Values reflect monthly average energy consumption in kWh
        - Detection flags (0/1) indicate presence of each appliance
        - Grid consumption represents total household electricity usage
        - Model confidence levels are not shown in this simplified output
        """)

        # Add interactive filtering directly with the first dataframe
        st.subheader("Sample Model Output with Interactive Filtering")

        # Add filter controls in a more compact format
        filter_cols = st.columns(4)

        with filter_cols[0]:
            ev_filter = st.selectbox("EV Charging", ["Any", "Present", "Not Present"])

        with filter_cols[1]:
            ac_filter = st.selectbox("Air Conditioning", ["Any", "Present", "Not Present"])

        with filter_cols[2]:
            pv_filter = st.selectbox("Solar PV", ["Any", "Present", "Not Present"])

        with filter_cols[3]:
            wh_filter = st.selectbox("Water Heater", ["Any", "Present", "Not Present"])

        # Apply filters
        filtered_df = df.copy()

        if ev_filter == "Present":
            filtered_df = filtered_df[filtered_df['ev detected'] == 1]
        elif ev_filter == "Not Present":
            filtered_df = filtered_df[filtered_df['ev detected'] == 0]

        if ac_filter == "Present":
            filtered_df = filtered_df[filtered_df['ac detected'] == 1]
        elif ac_filter == "Not Present":
            filtered_df = filtered_df[filtered_df['ac detected'] == 0]

        if pv_filter == "Present":
            filtered_df = filtered_df[filtered_df['pv detected'] == 1]
        elif pv_filter == "Not Present":
            filtered_df = filtered_df[filtered_df['pv detected'] == 0]

        if wh_filter == "Present":
            filtered_df = filtered_df[filtered_df['water heater detected'] == 1]
        elif wh_filter == "Not Present":
            filtered_df = filtered_df[filtered_df['water heater detected'] == 0]

        # Display filtered dataframe with record count
        st.dataframe(filtered_df, use_container_width=True)
        st.caption(f"Showing {len(filtered_df)} of {len(df)} homes")

        # Create two columns for the interactive plots
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Appliance Presence in Housing Portfolio")
            
            # Calculate presence percentages
            appliance_presence = {
                'EV Charging': filtered_df['ev detected'].sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0,
                'Air Conditioning': filtered_df['ac detected'].sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0,
                'Solar PV': filtered_df['pv detected'].sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0,
                'Water Heater': filtered_df['water heater detected'].sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
            }
            
            # Create interactive bar chart using Plotly with team colors
            fig1 = px.bar(
                x=list(appliance_presence.keys()),
                y=list(appliance_presence.values()),
                labels={'x': 'Appliance Type', 'y': 'Percentage of Homes (%)'},
                color=list(appliance_presence.keys()),
                color_discrete_map={
                        'EV Charging': primary_purple,
                        'Air Conditioning': green,
                        'Solar PV': cream,
                        'Water Heater': salmon
                },
                text=[f"{val:.1f}%" for val in appliance_presence.values()]
            )
            
            # Update layout with team colors - now with white background
            fig1.update_layout(
                showlegend=False,
                xaxis_title="Appliance Type",
                yaxis_title="Percentage of Homes (%)",
                yaxis_range=[0, 100],
                margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor=white,
                    plot_bgcolor=white,
                    font=dict(color=dark_purple)
            )
            
            fig1.update_traces(textposition='outside', textfont=dict(color=dark_purple))
            fig1.update_xaxes(showgrid=False, gridcolor=light_purple, tickfont=dict(color=dark_purple))
            fig1.update_yaxes(showgrid=True, gridcolor=light_purple, tickfont=dict(color=dark_purple))
            
            # Display the plot
            st.plotly_chart(fig1, use_container_width=True)
            
            # Add interactive details
            with st.expander("About Appliance Presence"):
                st.markdown("""
                This chart shows the percentage of homes in the sample where each appliance type was detected.
                
                - **Air Conditioning**: Most commonly detected appliance
                - **Solar PV**: Shows significant renewable adoption
                - **EV Charging**: Indicates electric vehicle ownership
                - **Water Heater**: Least commonly detected in this sample
                
                Detection is based on energy signature patterns identified by the disaggregation model.
                """)

        with col2:
            st.subheader("Total Energy Distribution by Type")
            
            # Calculate disaggregated appliance total
            disaggregated_total = (filtered_df['ev charging (kWh)'].sum() + 
                                  filtered_df['air conditioning (kWh)'].sum() + 
                                  filtered_df['water heater (kWh)'].sum())
            
            # Calculate "Other Consumption" by subtracting known appliances from grid total
            other_consumption = filtered_df['grid (kWh)'].sum() - disaggregated_total
            other_consumption = max(0, other_consumption)  # Ensure it's not negative
            
            energy_totals = {
                'EV Charging': filtered_df['ev charging (kWh)'].sum(),
                'Air Conditioning': filtered_df['air conditioning (kWh)'].sum(),
                'Water Heater': filtered_df['water heater (kWh)'].sum(),
                'Other Consumption': other_consumption
            }
            
            # Create interactive pie chart using Plotly with team colors
            fig2 = px.pie(
                values=list(energy_totals.values()),
                names=list(energy_totals.keys()),
                color=list(energy_totals.keys()),
                color_discrete_map={
                        'EV Charging': primary_purple,
                        'Air Conditioning': green,
                        'Water Heater': salmon,
                        'Other Consumption': light_purple
                },
                hole=0.4
            )
            
            # Update layout with team colors - now with white background
            fig2.update_layout(
                legend_title="Energy Type",
                margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor=white,
                    plot_bgcolor=white,
                    font=dict(color=dark_purple),
                    legend=dict(font=dict(color=dark_purple))
            )
            
            fig2.update_traces(
                textinfo='percent+label',
                    hovertemplate='%{label}<br>%{value:.1f} kWh<br>%{percent}',
                    textfont=dict(color=dark_gray)  # Darker text for better contrast
            )
            
            # Display the plot
            st.plotly_chart(fig2, use_container_width=True)
            
            # Add interactive details
            with st.expander("About Energy Distribution"):
                st.markdown("""
                    This chart shows how the total energy consumption is distributed across different appliance types.
                
                - **Air Conditioning**: Typically accounts for significant consumption
                - **EV Charging**: Can be a major energy consumer when present
                - **Water Heater**: Generally smaller portion of total energy use
                    - **Other Consumption**: Remaining grid usage not attributed to the three main appliances
                
                Understanding this distribution helps identify the highest impact areas for efficiency improvements.
                """)

        # Add summary metrics
        st.markdown(f"""
        <style>
            .metric-container {{
                background-color: {primary_purple};
                border-radius: 10px;
                padding: 15px 15px;
                margin: 10px 0;
                border: 2px solid {primary_purple};
            }}
        </style>
        """, unsafe_allow_html=True)

        st.subheader("Key Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric(
                label="Total Homes",
                value=len(filtered_df),
                delta=f"{len(filtered_df) - len(df)}" if len(filtered_df) != len(df) else None,
                help="Number of households in the filtered dataset"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with metric_col2:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            avg_grid = filtered_df['grid (kWh)'].mean() if len(filtered_df) > 0 else 0
            st.metric(
                label="Avg. Grid Consumption",
                value=f"{avg_grid:.1f} kWh",
                help="Average total electricity consumption per home"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with metric_col3:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            pv_homes = filtered_df[filtered_df['pv detected'] == 1]
            pv_avg = pv_homes['solar production (kWh)'].mean() if len(pv_homes) > 0 else 0
            st.metric(
                label="Avg. Solar Production",
                value=f"{pv_avg:.1f} kWh",
                help="Average solar production for homes with PV systems"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with metric_col4:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            # Percentage of consumption identified by model
            total_grid = filtered_df['grid (kWh)'].sum()
            total_identified = disaggregated_total
            pct_identified = (total_identified / total_grid * 100) if total_grid > 0 else 0
            
            st.metric(
                label="Consumption Identified",
                value=f"{pct_identified:.1f}%",
                help="Percentage of total grid consumption attributed to specific appliances"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Add footer with primary purple color
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align:center; color:{primary_purple}; padding: 10px; border-radius: 5px;">
            This sample output demonstrates the type of insights available from the disaggregation model. 
            In a full deployment, thousands of households would be analyzed to provide statistically significant patterns and trends.
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        st.info("Please make sure the 'disagg_sample.csv' file is in the same directory as this script.")

elif page == "Performance Metrics":
    # Performance Metrics page
    # Just one title, with a more comprehensive subheader
    st.title("NILM Performance Dashboard")
    st.subheader(f"Device Detection Performance Analysis")

elif page == "Interactive Map":
    # Interactive Map page
    st.title("NILM Deployment Map")
    st.subheader("Geographic Distribution of Homes with Smart Devices")
    
    # Description
    st.markdown("""
    This map shows the geographic distribution of homes equipped with NILM-detected devices.
    **Click on a state** to zoom in and see individual homes with EV chargers, AC units, and solar panels.
    """)
    
    # Create function to generate mock data for US states
    @st.cache_data
    def generate_geo_data():
        # US states with coordinates (approximate centers)
        states_data = {
            'CA': {'name': 'California', 'lat': 36.7783, 'lon': -119.4179, 'zoom': 6},
            'TX': {'name': 'Texas', 'lat': 31.9686, 'lon': -99.9018, 'zoom': 6},
            'NY': {'name': 'New York', 'lat': 42.1657, 'lon': -74.9481, 'zoom': 7},
            'FL': {'name': 'Florida', 'lat': 27.6648, 'lon': -81.5158, 'zoom': 6},
            'IL': {'name': 'Illinois', 'lat': 40.6331, 'lon': -89.3985, 'zoom': 7},
            'PA': {'name': 'Pennsylvania', 'lat': 41.2033, 'lon': -77.1945, 'zoom': 7},
            'OH': {'name': 'Ohio', 'lat': 40.4173, 'lon': -82.9071, 'zoom': 7},
            'MA': {'name': 'Massachusetts', 'lat': 42.4072, 'lon': -71.3824, 'zoom': 8},
            'WA': {'name': 'Washington', 'lat': 47.7511, 'lon': -120.7401, 'zoom': 7}
        }
        
        # Generate stats for each state
        for state_code in states_data:
            state = states_data[state_code]
            state['total_homes'] = random.randint(150, 500)
            state['ev_homes'] = random.randint(30, int(state['total_homes'] * 0.3))
            state['ac_homes'] = random.randint(int(state['total_homes'] * 0.5), int(state['total_homes'] * 0.9))
            state['pv_homes'] = random.randint(20, int(state['total_homes'] * 0.25))
        
        def generate_households(state_code, count=100):
            """Generate mock household data within a state"""
            state = states_data[state_code]
            households = []
            
            # Define the spread of points (in degrees)
            lat_spread = 1.5
            lon_spread = 1.5
            
            for i in range(count):
                # Randomly place homes around the state center
                lat = state['lat'] + (random.random() - 0.5) * lat_spread
                lon = state['lon'] + (random.random() - 0.5) * lon_spread
                
                # Assign devices randomly but weighted by state percentages
                has_ev = random.random() < (state['ev_homes'] / state['total_homes'])
                has_ac = random.random() < (state['ac_homes'] / state['total_homes'])
                has_pv = random.random() < (state['pv_homes'] / state['total_homes'])
                
                # Ensure at least one device is present
                if not (has_ev or has_ac or has_pv):
                    # Assign at least one device
                    device_type = random.choice(['ev', 'ac', 'pv'])
                    if device_type == 'ev': has_ev = True
                    elif device_type == 'ac': has_ac = True
                    else: has_pv = True
                
                # Create household data
                household = {
                    'id': f"{state_code}-{i+1}",
                    'lat': lat,
                    'lon': lon,
                    'has_ev': has_ev,
                    'has_ac': has_ac,
                    'has_pv': has_pv,
                    'energy_consumption': random.randint(20, 100),  # kWh per day
                    'state': state_code
                }
                households.append(household)
            
            return households
        
        # Generate households for each state
        all_households = []
        for state_code in states_data:
            state_households = generate_households(state_code, states_data[state_code]['total_homes'])
            all_households.extend(state_households)
        
        return states_data, all_households
    
    # Load GeoJSON data for US states
    @st.cache_data
    def load_us_geojson():
        # Use actual GeoJSON data from an external source for more accurate state boundaries
        import requests
        
        try:
            # Try to fetch real US state boundary data
            response = requests.get("https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json")
            us_states = response.json()
            
            # Filter to only include our selected states
            state_codes = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'MA', 'WA']
            us_states['features'] = [f for f in us_states['features'] 
                                    if f['id'] in state_codes]
            
            # Add our custom density data to real boundaries
            state_density = {
                'CA': 241.7,
                'TX': 103.4,
                'NY': 417.2,
                'FL': 378.5,
                'IL': 231.1,
                'PA': 284.3,
                'OH': 282.5,
                'MA': 863.8,
                'WA': 107.9
            }
            
            # Update properties to include our code format and density data
            for feature in us_states['features']:
                state_id = feature['id']
                feature['properties']['code'] = state_id
                feature['properties']['density'] = state_density.get(state_id, 0)
            
            return us_states
            
        except Exception as e:
            # Fallback to simplified geometry if network fetch fails
            st.warning(f"Couldn't fetch US state boundaries, using simplified fallback data.")
            
            # Use a simplified version as backup
            us_states = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "California", "code": "CA", "density": 241.7},
                        "geometry": {"type": "Polygon", "coordinates": [[[-124.3, 32.5], [-124.3, 42.0], [-114.1, 42.0], [-114.1, 32.5], [-124.3, 32.5]]]}
                    },
                    {
                        "type": "Feature",
                        "properties": {"name": "Texas", "code": "TX", "density": 103.4},
                        "geometry": {"type": "Polygon", "coordinates": [[[-106.6, 25.8], [-106.6, 36.5], [-93.5, 36.5], [-93.5, 25.8], [-106.6, 25.8]]]}
                    },
                    {
                        "type": "Feature",
                        "properties": {"name": "New York", "code": "NY", "density": 417.2},
                        "geometry": {"type": "Polygon", "coordinates": [[[-79.8, 40.5], [-79.8, 45.0], [-71.8, 45.0], [-71.8, 40.5], [-79.8, 40.5]]]}
                    },
                    {
                        "type": "Feature",
                        "properties": {"name": "Florida", "code": "FL", "density": 378.5},
                        "geometry": {"type": "Polygon", "coordinates": [[[-87.6, 24.5], [-87.6, 31.0], [-80.0, 31.0], [-80.0, 24.5], [-87.6, 24.5]]]}
                    },
                    {
                        "type": "Feature",
                        "properties": {"name": "Illinois", "code": "IL", "density": 231.1},
                        "geometry": {"type": "Polygon", "coordinates": [[[-91.5, 37.0], [-91.5, 42.5], [-87.5, 42.5], [-87.5, 37.0], [-91.5, 37.0]]]}
                    },
                    {
                        "type": "Feature",
                        "properties": {"name": "Pennsylvania", "code": "PA", "density": 284.3},
                        "geometry": {"type": "Polygon", "coordinates": [[[-80.5, 39.7], [-80.5, 42.3], [-74.7, 42.3], [-74.7, 39.7], [-80.5, 39.7]]]}
                    },
                    {
                        "type": "Feature",
                        "properties": {"name": "Ohio", "code": "OH", "density": 282.5},
                        "geometry": {"type": "Polygon", "coordinates": [[[-84.8, 38.4], [-84.8, 42.0], [-80.5, 42.0], [-80.5, 38.4], [-84.8, 38.4]]]}
                    },
                    {
                        "type": "Feature",
                        "properties": {"name": "Massachusetts", "code": "MA", "density": 863.8},
                        "geometry": {"type": "Polygon", "coordinates": [[[-73.5, 41.2], [-73.5, 42.9], [-69.9, 42.9], [-69.9, 41.2], [-73.5, 41.2]]]}
                    },
                    {
                        "type": "Feature",
                        "properties": {"name": "Washington", "code": "WA", "density": 107.9},
                        "geometry": {"type": "Polygon", "coordinates": [[[-124.8, 45.5], [-124.8, 49.0], [-116.9, 49.0], [-116.9, 45.5], [-124.8, 45.5]]]}
                    }
                ]
            }
            
            return us_states
    
    # Load the data
    states_data, households = generate_geo_data()
    us_states_geojson = load_us_geojson()
    
    # Create filter controls in sidebar
    st.sidebar.markdown("### Map Filters")
    show_ev = st.sidebar.checkbox("Show EV Chargers", value=True)
    show_ac = st.sidebar.checkbox("Show AC Units", value=True)
    show_pv = st.sidebar.checkbox("Show Solar Panels", value=True)
    
    # Get state from URL parameter if available
    params = st.query_params
    url_state = params.get("state", [""])[0]
    
    # Set the default selected state to empty string for the map overview
    selected_state = ""

    # Check if we have a valid state from URL parameters
    if url_state in states_data:
        selected_state = url_state
        # Hide the state selection dropdown when a state is selected via URL
        st.markdown("""
        <style>
        /* Hide the state selection dropdown when viewing a state detail */
        div[data-testid="stSelectbox"] {display: none !important;}
        </style>
        """, unsafe_allow_html=True)
    else:
        # If no valid state in URL, use the dropdown but keep it hidden on the map page
        selected_state_dropdown = st.selectbox(
            "Select State to View",
            options=list(states_data.keys()),
            format_func=lambda x: states_data[x]['name'],
            index=0
        )
        
        # Only use the dropdown value if we're not in map overview mode
        if selected_state_dropdown:
            selected_state = selected_state_dropdown
    
    # Filter households by selected state and device types
    filtered_households = [
        h for h in households if 
        h['state'] == selected_state and
        ((show_ev and h['has_ev']) or 
         (show_ac and h['has_ac']) or 
         (show_pv and h['has_pv']))
    ]
    
    # Display stats for selected state
    if selected_state:
        # Create a row with columns for the back button and a note
        back_cols = st.columns([1, 3])
        
        # Add a Streamlit-based back button at the top in the first column
        with back_cols[0]:
            back_button = st.button("← Back to Map Overview", type="primary", use_container_width=True)
            if back_button:
                st.query_params.clear()
                st.rerun()

        # Add a note about the map navigation in the second column
        with back_cols[1]:
            st.markdown("""
            <div style="margin-top: 5px; color: #666; font-style: italic;">
                <small>Note: You can also use the blue "Back to Map" button in the top-left corner of the map below.</small>
            </div>
            """, unsafe_allow_html=True)
            
        state = states_data[selected_state]
        st.markdown(f"### {state['name']} Statistics")
        
        # Create metrics for the selected state
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Homes", f"{state['total_homes']:,}")
        with col2:
            st.metric("Homes with EV Chargers", f"{state['ev_homes']:,}", 
                    f"{state['ev_homes']/state['total_homes']*100:.1f}%")
        with col3:
            st.metric("Homes with AC Units", f"{state['ac_homes']:,}", 
                    f"{state['ac_homes']/state['total_homes']*100:.1f}%")
        with col4:
            st.metric("Homes with Solar Panels", f"{state['pv_homes']:,}", 
                    f"{state['pv_homes']/state['total_homes']*100:.1f}%")
    else:
        # If we're on the overview map, show a message prompting to select a state
        st.markdown("""
        <div style="text-align: center; padding: 20px; background-color: rgba(81, 93, 154, 0.05); border-radius: 5px; margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #515D9A;">Select a State on the Map</h3>
            <p>Click directly on any state in the map below to view detailed statistics and device distribution.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create map
    st.markdown("### Interactive Map")
    
    # Update the description to explain the interactivity clearly
    st.markdown("""
    <div style="background-color: rgba(81, 93, 154, 0.1); padding: 15px; border-radius: 5px; margin-bottom: 15px;">
        <h4 style="margin-top: 0;">How to Use This Map</h4>
        <p><strong>🖱️ Click directly on any state</strong> to zoom in and view homes with smart devices.</p>
        <p>Once zoomed in, you can see individual homes with:</p>
        <ul>
            <li>🔌 EV chargers</li>
            <li>❄️ AC units</li>
            <li>☀️ Solar panels</li>
        </ul>
        <p>Use the <strong>Back to Map</strong> button to return to the overview.</p>
    </div>
    <p><em><strong>DISCLAIMER:</strong> Currently, the map was generated using synthetic data, for testing purposes.</em></p>
    """, unsafe_allow_html=True)
    
    # Create two types of maps: overview and detail
    if selected_state == "":
        # Overview map of all states
        m = folium.Map(
            location=[39.8283, -98.5795],  # Center of US
            zoom_start=4,
            tiles="CartoDB positron"
        )

        # Add satellite view tile layer
        folium.TileLayer('Esri_WorldImagery', name='Satellite View', attr='Esri').add_to(m)

        # Add GeoJSON states layer with click functionality
        style_function = lambda feature: {
            'fillColor': primary_purple,
            'color': 'white',
            'weight': 1,
            'fillOpacity': 0.5,
        }
        
        highlight_function = lambda feature: {
            'fillColor': light_purple,
            'color': 'white',
            'weight': 3,
            'fillOpacity': 0.7,
        }
        
        # Custom JavaScript for handling clicks on states
        click_script = """
        function (feature, layer) {
            // Make sure this layer is clickable
            layer.options.interactive = true;
            
            // Create a function for navigation that works for both direct and popup clicks
            function navigateToState(stateCode) {
                console.log("Navigating to state: " + stateCode);
                // Use window.location.search instead of href to avoid opening a new page
                window.location.search = "state=" + stateCode;
            }
            
            // Add a direct click handler
            layer.on({
                mouseover: function (e) {
                    var layer = e.target;
                    layer.setStyle({
                        fillOpacity: 0.7,
                        fillColor: '#B8BCF3',
                        weight: 3,
                        color: 'white'
                    });
                    
                    if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {
                        layer.bringToFront();
                    }
                },
                mouseout: function (e) {
                    var layer = e.target;
                    layer.setStyle({
                        fillOpacity: 0.5,
                        fillColor: '#515D9A',
                        weight: 1,
                        color: 'white'
                    });
                },
                click: function (e) {
                    // Visual feedback on click
                    var layer = e.target;
                    layer.setStyle({
                        fillOpacity: 0.9,
                        fillColor: '#515D9A',
                        weight: 4,
                        color: '#FFFFFF'
                    });
                    
                    // Get the state code
                    var stateCode = feature.properties.code || feature.id;
                    console.log("Clicked on state: " + stateCode);
                    
                    // Add a small delay for visual feedback before navigating
                    setTimeout(function() {
                        navigateToState(stateCode);
                    }, 300);
                }
            });
            
            // Also ensure any popup clicks navigate correctly
            layer.on('popupopen', function() {
                setTimeout(function() {
                    var popups = document.getElementsByClassName('leaflet-popup-content');
                    if (popups.length > 0) {
                        var stateCode = feature.properties.code || feature.id;
                        popups[0].addEventListener('click', function() {
                            navigateToState(stateCode);
                        });
                    }
                }, 100);
            });
        }
        """
        
        # Add the GeoJSON layer with click handler
        geojson_tooltip_fields = ['name']
        if 'density' in us_states_geojson['features'][0]['properties']:
            geojson_tooltip_fields.append('density')
            
        folium.GeoJson(
            us_states_geojson,
            name="US States",
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.GeoJsonTooltip(
                fields=geojson_tooltip_fields,
                aliases=['State:'] + (['Population Density:'] if len(geojson_tooltip_fields) > 1 else []),
                labels=True,
                sticky=True,
                style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.2);")
            ),
            popup=folium.GeoJsonPopup(
                fields=['name'],
                aliases=['Click to view devices in:'],
                style=("background-color: white; color: #333333; font-family: arial; font-size: 14px; padding: 10px; border-radius: 5px; font-weight: bold;")
            ),
            script=click_script
        ).add_to(m)
        
        # Add state border lines for clarity
        folium.GeoJson(
            us_states_geojson,
            name="State Borders",
            style_function=lambda x: {
                'color': '#666666',
                'weight': 2,
                'fillOpacity': 0
            },
            tooltip=None
        ).add_to(m)
        
        # Add state markers with statistics
        for state_code, state_info in states_data.items():
            # Create popup content with statistics
            popup_content = f"""
            <div style="width: 200px;">
                <h4>{state_info['name']}</h4>
                <b>Total Homes:</b> {state_info['total_homes']}<br>
                <b>Homes with EV:</b> {state_info['ev_homes']} ({state_info['ev_homes']/state_info['total_homes']*100:.1f}%)<br>
                <b>Homes with AC:</b> {state_info['ac_homes']} ({state_info['ac_homes']/state_info['total_homes']*100:.1f}%)<br>
                <b>Homes with PV:</b> {state_info['pv_homes']} ({state_info['pv_homes']/state_info['total_homes']*100:.1f}%)<br>
                <a href="javascript:void(0);" onclick="window.location.search='state={state_code}'" style="color: #515D9A; font-weight: bold;">Click to view details</a>
            </div>
            """
            
            # Add marker
            folium.Marker(
                location=[state_info['lat'], state_info['lon']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(icon="info-sign", prefix="fa", color="purple"),
                tooltip=f"Click for {state_info['name']} statistics"
            ).add_to(m)
        
    else:
        # Detailed map for selected state
        m = folium.Map(
            location=[state['lat'], state['lon']], 
            zoom_start=state['zoom'],
            tiles="CartoDB positron"
        )

        # Add satellite view tile layer
        folium.TileLayer('Esri_WorldImagery', name='Satellite View', attr='Esri').add_to(m)

        # Add a back button to the overview map with improved styling
        back_button_html = '''
        <div id="back-button" style="position: absolute; 
                    top: 10px; left: 10px; width: 140px; height: 40px; 
                    z-index:9999999; font-size:15px; background-color:#FFFFFF; 
                    border-radius:6px; padding: 8px; box-shadow: 0 3px 12px rgba(0,0,0,0.4);
                    text-align:center; transition: all 0.2s ease; cursor:pointer;
                    border: 3px solid #515D9A;">
            <a href="javascript:void(0);" id="back-link" style="color:#515D9A; text-decoration:none; font-weight:bold; display:flex; align-items:center; justify-content:center; width: 100%; height: 100%;">
                <i class="fa fa-arrow-left" style="margin-right: 8px; font-size: 18px;"></i> Back to Map
            </a>
        </div>
        
        <script>
        // Make sure the back button works by adding multiple event handlers
        setTimeout(function() {
            try {
                var backButton = document.getElementById('back-button');
                var backLink = document.getElementById('back-link');
                
                function navigateToOverview() {
                    // Use window.location.search instead of href to avoid opening a new page
                    window.location.search = '';
                }
                
                if (backButton) {
                    // Direct click on button area
                    backButton.addEventListener('click', function(e) {
                        e.preventDefault();
                        navigateToOverview();
                    });
                    
                    // Hover effects
                    backButton.addEventListener('mouseover', function() {
                        this.style.backgroundColor = "#f0f5ff";
                        this.style.boxShadow = "0 4px 15px rgba(0,0,0,0.4)";
                        this.style.transform = "translateY(-2px)";
                    });
                    
                    backButton.addEventListener('mouseout', function() {
                        this.style.backgroundColor = "#FFFFFF";
                        this.style.boxShadow = "0 3px 12px rgba(0,0,0,0.4)";
                        this.style.transform = "translateY(0)";
                    });
                    
                    // Active effect
                    backButton.addEventListener('mousedown', function() {
                        this.style.transform = "scale(0.97)";
                    });
                    
                    backButton.addEventListener('mouseup', function() {
                        this.style.transform = "scale(1)";
                    });
                }
                
                if (backLink) {
                    // Link click event
                    backLink.addEventListener('click', function(e) {
                        e.preventDefault();
                        navigateToOverview();
                    });
                }
                
                // Ensure map controls don't overlap with button
                var leftControls = document.querySelector('.leaflet-top.leaflet-left');
                if (leftControls) {
                    leftControls.style.top = "60px";
                }
                
                // Add a pulsing effect to draw attention to the button
                setTimeout(function() {
                    if (backButton) {
                        backButton.style.transform = "scale(1.05)";
                        setTimeout(function() {
                            backButton.style.transform = "scale(1)";
                        }, 300);
                    }
                }, 1000);
                
            } catch(e) {
                console.error("Error setting up back button:", e);
            }
        }, 300);
        </script>
        '''
        m.get_root().html.add_child(folium.Element(back_button_html))
        
        # Add state boundaries to the detail view
        # Create a GeoJSON layer just for this state (if we had full US GeoJSON)
        state_feature = None
        
        # First try to find the state by code property
        try:
            state_feature = next((feature for feature in us_states_geojson['features'] 
                              if feature['properties'].get('code') == selected_state), None)
        except:
            pass
            
        # If not found, try to find by id (which is used in the external GeoJSON)
        if not state_feature:
            try:
                state_feature = next((feature for feature in us_states_geojson['features'] 
                                  if feature.get('id') == selected_state), None)
            except:
                pass
        
        # If we found the state feature, add it to the map
        if state_feature:
            # Add just this state's boundary
            folium.GeoJson(
                {"type": "FeatureCollection", "features": [state_feature]},
                name=f"{state['name']} Boundary",
                style_function=lambda x: {
                    'color': primary_purple,
                    'weight': 3,
                    'fillOpacity': 0.1,
                    'fillColor': light_purple
                }
            ).add_to(m)
        
        # Create marker cluster for the households
        marker_cluster = MarkerCluster().add_to(m)
        
        # Device count summary
        ev_count = sum(1 for h in filtered_households if h['has_ev'])
        ac_count = sum(1 for h in filtered_households if h['has_ac'])
        pv_count = sum(1 for h in filtered_households if h['has_pv'])
        
        # Add state center marker with summary
        summary_popup = f"""
        <div style="width: 240px; padding: 10px;">
            <h4 style="margin-top: 0; color: #515D9A; border-bottom: 2px solid #515D9A; padding-bottom: 5px; margin-bottom: 10px;">{state['name']} Summary</h4>
            <div style="display: flex; margin-bottom: 5px;">
                <div style="width: 30px; text-align: center;"><i class="fa fa-home" style="color: #515D9A;"></i></div>
                <div><b>Total Homes:</b> {len(filtered_households)}</div>
            </div>
            <div style="display: flex; margin-bottom: 5px;">
                <div style="width: 30px; text-align: center;"><i class="fa fa-plug" style="color: #515D9A;"></i></div>
                <div><b>EV Chargers:</b> {ev_count} ({ev_count/len(filtered_households)*100:.1f}%)</div>
            </div>
            <div style="display: flex; margin-bottom: 5px;">
                <div style="width: 30px; text-align: center;"><i class="fa fa-snowflake-o" style="color: #515D9A;"></i></div>
                <div><b>AC Units:</b> {ac_count} ({ac_count/len(filtered_households)*100:.1f}%)</div>
            </div>
            <div style="display: flex;">
                <div style="width: 30px; text-align: center;"><i class="fa fa-sun-o" style="color: #515D9A;"></i></div>
                <div><b>Solar Panels:</b> {pv_count} ({pv_count/len(filtered_households)*100:.1f}%)</div>
            </div>
        </div>
        """
        
        # Add a state information panel to the map
        info_panel_html = f"""
        <div id="state-info-panel" style="
            position: absolute; 
            top: 10px; 
            right: 10px; 
            width: 260px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.2);
            padding: 15px;
            z-index: 1000;
            font-family: Arial, sans-serif;
        ">
            <h3 style="margin-top: 0; color: #515D9A; border-bottom: 2px solid #515D9A; padding-bottom: 5px;">{state['name']}</h3>
            <div style="margin-bottom: 15px;">
                <div style="font-weight: bold; margin-bottom: 5px;">Device Distribution:</div>
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <i class="fa fa-plug" style="color: blue; margin-right: 10px; width: 15px;"></i>
                    <div style="flex-grow: 1;">EV Chargers</div>
                    <div style="font-weight: bold;">{ev_count}/{len(filtered_households)}</div>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <i class="fa fa-snowflake-o" style="color: orange; margin-right: 10px; width: 15px;"></i>
                    <div style="flex-grow: 1;">AC Units</div>
                    <div style="font-weight: bold;">{ac_count}/{len(filtered_households)}</div>
                </div>
                <div style="display: flex; align-items: center;">
                    <i class="fa fa-sun-o" style="color: green; margin-right: 10px; width: 15px;"></i>
                    <div style="flex-grow: 1;">Solar Panels</div>
                    <div style="font-weight: bold;">{pv_count}/{len(filtered_households)}</div>
                </div>
            </div>
            <div style="font-size: 12px; font-style: italic; color: #666;">
                Click on markers to view device details for each home
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(info_panel_html))
        
        folium.Marker(
            [state['lat'], state['lon']],
            popup=folium.Popup(summary_popup, max_width=300),
            icon=folium.Icon(color="purple", icon="info-sign", prefix="fa"),
            tooltip=f"{state['name']} Summary"
        ).add_to(m)
        
        # Add markers for each household
        for house in filtered_households:
            # Determine marker icon based on devices
            if house['has_ev'] and show_ev:
                icon_color = "blue"
                icon_name = "plug"
            elif house['has_pv'] and show_pv:
                icon_color = "green"
                icon_name = "sun"
            else:
                icon_color = "orange"
                icon_name = "home"
            
            # Create popup content
            popup_content = f"""
            <div style="min-width: 220px; padding: 10px;">
                <h4 style="margin-top: 0; color: #515D9A; border-bottom: 2px solid #515D9A; padding-bottom: 5px;">Home {house['id']}</h4>
                
                <div style="margin-top: 10px; margin-bottom: 10px;">
                    <div style="font-weight: bold; margin-bottom: 5px;">Installed Devices:</div>
                    <div style="display: flex; margin-bottom: 5px; align-items: center;">
                        <div style="width: 20px; text-align: center; margin-right: 5px;">
                            <i class="fa {'fa-check-circle' if house['has_ev'] else 'fa-times-circle'}" 
                               style="color: {'#3366cc' if house['has_ev'] else '#cccccc'}; font-size: 16px;"></i>
                        </div>
                        <div>EV Charger</div>
                    </div>
                    <div style="display: flex; margin-bottom: 5px; align-items: center;">
                        <div style="width: 20px; text-align: center; margin-right: 5px;">
                            <i class="fa {'fa-check-circle' if house['has_ac'] else 'fa-times-circle'}" 
                               style="color: {'#ff9900' if house['has_ac'] else '#cccccc'}; font-size: 16px;"></i>
                        </div>
                        <div>AC Unit</div>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 20px; text-align: center; margin-right: 5px;">
                            <i class="fa {'fa-check-circle' if house['has_pv'] else 'fa-times-circle'}" 
                               style="color: {'#66cc66' if house['has_pv'] else '#cccccc'}; font-size: 16px;"></i>
                        </div>
                        <div>Solar Panels</div>
                    </div>
                </div>
                
                <div style="margin-top: 15px; display: flex; align-items: center;">
                    <div style="width: 20px; text-align: center; margin-right: 5px;">
                        <i class="fa fa-bolt" style="color: #515D9A;"></i>
                    </div>
                    <div><b>Daily Energy:</b> {house['energy_consumption']} kWh</div>
                </div>
            </div>
            """
            
            # Determine tooltip text based on devices
            if house['has_ev'] and house['has_ac'] and house['has_pv']:
                tooltip_text = f"Home with EV, AC & Solar"
            elif house['has_ev'] and house['has_ac']:
                tooltip_text = f"Home with EV & AC"
            elif house['has_ev'] and house['has_pv']:
                tooltip_text = f"Home with EV & Solar"
            elif house['has_ac'] and house['has_pv']:
                tooltip_text = f"Home with AC & Solar"
            elif house['has_ev']:
                tooltip_text = f"Home with EV Charger"
            elif house['has_ac']:
                tooltip_text = f"Home with AC Unit"
            elif house['has_pv']:
                tooltip_text = f"Home with Solar Panels"
            else:
                tooltip_text = f"Home {house['id']}"
            
            # Add marker
            folium.Marker(
                location=[house['lat'], house['lon']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa'),
                tooltip=tooltip_text
            ).add_to(marker_cluster)
    
    # Add a custom layer control
    folium.LayerControl().add_to(m)
    
    # Add a legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 180px; height: auto;
                border:2px solid grey; z-index:9999; background-color:white;
                padding: 10px; font-size:14px; border-radius:6px; box-shadow: 0 0 10px rgba(0,0,0,0.2);">
        <div style="margin-bottom: 5px;"><strong>Device Types</strong></div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <i class="fa fa-circle" style="color:blue; margin-right: 5px;"></i> EV Charger
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <i class="fa fa-circle" style="color:green; margin-right: 5px;"></i> Solar Panels
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <i class="fa fa-circle" style="color:orange; margin-right: 5px;"></i> AC Unit
        </div>
        <div style="display: flex; align-items: center;">
            <i class="fa fa-circle" style="color:purple; margin-right: 5px;"></i> State Summary
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Display the map
    folium_static(m, width=1000, height=600)
    
    
    # Display data table for the filtered households if a state is selected
    if selected_state:
        st.markdown("### Filtered Homes Data")
        
        # Convert filtered households to DataFrame for display
        df_houses = pd.DataFrame([
            {
                'Home ID': h['id'],
                'Has EV Charger': '✓' if h['has_ev'] else '',
                'Has AC Unit': '✓' if h['has_ac'] else '',
                'Has Solar Panels': '✓' if h['has_pv'] else '',
                'Energy Consumption (kWh/day)': h['energy_consumption']
            }
            for h in filtered_households
        ])
        
        # Show the data table
        st.dataframe(df_houses)
        
        # Add download button for the data
        csv = df_houses.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"{state['name']}_homes_data.csv",
            mime="text/csv",
        )
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; color:{primary_purple}; padding: 10px; border-radius: 5px;">
        This interactive map shows the geographic distribution of homes with different smart devices.
        Click on a state to zoom in and explore homes with EV chargers, AC units, and solar panels.
    </div>
    """, unsafe_allow_html=True)
