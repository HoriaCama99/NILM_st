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

# Add page selection at the top (only two options)
page = st.sidebar.radio("Select Page", ["Sample Output", "Performance Metrics", "Interactive Map"])

# Display banner on both pages
try:
    # Display banner image
    banner_image = Image.open(banner_path)
    st.image(banner_image, use_container_width=True)
except Exception as e:
    st.warning(f"Banner image not found at {banner_path}. Please update the path in the code.")

if page == "Sample Output":
    # Convert the sample data to a DataFrame
    df = pd.read_csv('disagg_sample.csv') 

    # Page title and introduction
    st.title("Energy Disaggregation Model: Sample Output")

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

    # Geographic Energy Insights Map
    st.subheader("Geographical Energy Insights")
    
    # Mock dataset for geographical visualization
    @st.cache_data
    def generate_geo_data():
        import random
        import datetime
        
        # Define states and regions
        regions = {
            'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
            'Southeast': ['DE', 'MD', 'VA', 'WV', 'KY', 'NC', 'SC', 'TN', 'GA', 'FL', 'AL', 'MS', 'AR', 'LA'],
            'Midwest': ['OH', 'MI', 'IN', 'IL', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
            'Southwest': ['OK', 'TX', 'NM', 'AZ'],
            'West': ['CO', 'WY', 'MT', 'ID', 'WA', 'OR', 'UT', 'NV', 'CA', 'AK', 'HI']
        }
        
        # Create a base for our mock dataset
        geo_data = []
        
        # Define time periods - will create data for past 24 months
        current_date = datetime.datetime.now()
        start_date = current_date - datetime.timedelta(days=24*30)  # approx 24 months back
        time_periods = []
        
        # Generate monthly time periods
        for i in range(24):
            month_date = start_date + datetime.timedelta(days=30*i)
            time_periods.append({
                'date': month_date,
                'year': month_date.year,
                'month': month_date.month,
                'month_name': month_date.strftime('%b'),
                'quarter': (month_date.month-1)//3 + 1,
                'period_label': month_date.strftime('%b %Y')
            })
        
        # Set region-specific patterns
        base_region_characteristics = {
            'Northeast': {
                'grid_range': (800, 1200),  # Higher consumption due to heating
                'solar_range': (100, 350),   # Moderate solar production
                'ev_range': (150, 400),      # Higher EV adoption 
                'ac_range': (100, 300),      # Lower AC due to climate
                'water_heater_range': (150, 300),  # Higher water heating needs
                'pv_adoption_range': (15, 40),     # Moderate solar adoption
                'ev_adoption_range': (15, 45)      # Higher EV adoption
            },
            'Southeast': {
                'grid_range': (900, 1400),   # Higher consumption due to AC
                'solar_range': (200, 450),   # Good solar production
                'ev_range': (50, 250),       # Lower EV adoption
                'ac_range': (250, 500),      # Higher AC usage
                'water_heater_range': (100, 200),  # Lower water heating needs
                'pv_adoption_range': (10, 35),     # Moderate solar adoption
                'ev_adoption_range': (5, 25)       # Lower EV adoption
            },
            'Midwest': {
                'grid_range': (700, 1100),   # Moderate consumption
                'solar_range': (50, 250),    # Lower solar production
                'ev_range': (50, 200),       # Moderate-low EV adoption
                'ac_range': (150, 350),      # Moderate AC usage
                'water_heater_range': (120, 250),  # Moderate water heating
                'pv_adoption_range': (5, 25),      # Lower solar adoption
                'ev_adoption_range': (8, 30)       # Moderate EV adoption
            },
            'Southwest': {
                'grid_range': (900, 1300),   # Higher consumption due to AC
                'solar_range': (300, 600),   # High solar production
                'ev_range': (100, 300),      # Moderate EV adoption
                'ac_range': (300, 550),      # Very high AC usage
                'water_heater_range': (80, 180),   # Lower water heating
                'pv_adoption_range': (25, 60),     # High solar adoption
                'ev_adoption_range': (10, 35)      # Moderate EV adoption
            },
            'West': {
                'grid_range': (600, 1000),   # Lower grid consumption (efficiency)
                'solar_range': (250, 550),   # High solar production
                'ev_range': (150, 450),      # High EV adoption
                'ac_range': (150, 350),      # Moderate AC usage
                'water_heater_range': (100, 200),  # Moderate water heating
                'pv_adoption_range': (20, 55),     # High solar adoption
                'ev_adoption_range': (15, 50)      # High EV adoption
            }
        }
        
        # Define seasonal adjustments by month
        seasonal_factors = {
            # Month -> (grid, solar, ev, ac, water_heater)
            1:  (1.2,  0.7,  0.9, 0.3, 1.4),  # January
            2:  (1.15, 0.8,  0.9, 0.3, 1.3),  # February
            3:  (1.0,  0.9,  0.95, 0.5, 1.2),  # March
            4:  (0.9,  1.0,  1.0, 0.7, 1.0),  # April
            5:  (0.8,  1.1,  1.0, 0.9, 0.9),  # May
            6:  (0.9,  1.2,  1.0, 1.3, 0.8),  # June
            7:  (1.0,  1.2,  1.1, 1.5, 0.7),  # July
            8:  (1.0,  1.15, 1.1, 1.4, 0.7),  # August
            9:  (0.9,  1.1,  1.05, 1.2, 0.8),  # September
            10: (0.85, 0.9,  1.0, 0.8, 0.9),  # October
            11: (0.95, 0.8,  0.95, 0.5, 1.1),  # November
            12: (1.1,  0.7,  0.9, 0.3, 1.3)   # December
        }
        
        # Define more pronounced yearly growth trends - make historical progression more noticeable
        # The factors represent multipliers for (grid, solar, ev, ac, water_heater, pv_adoption, ev_adoption, efficiency)
        yearly_growth = {
            # 0 = oldest data, 1 = middle data, 2 = newest data (2 years of data)
            0: (1.15, 0.7, 0.6, 1.0, 1.05, 0.5, 0.4, 0.7),  # 2 years ago: higher grid use, lower renewables & efficiency
            1: (1.05, 0.85, 0.8, 1.0, 1.0, 0.75, 0.7, 0.85),  # 1 year ago: transitioning
            2: (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)  # Current year: baseline
        }
        
        # Define innovation adoption tiers for states (which states adopt technologies first)
        # This creates more realistic patterns where certain states lead in adoption
        state_adoption_tiers = {
            'early_adopters': ['CA', 'NY', 'MA', 'WA', 'OR', 'CO', 'HI', 'VT'],
            'early_majority': ['CT', 'NJ', 'MD', 'IL', 'MN', 'AZ', 'NV', 'TX', 'FL', 'UT', 'NM'],
            'late_majority': ['PA', 'OH', 'MI', 'NC', 'SC', 'GA', 'VA', 'WI', 'IA', 'NH', 'ME', 'RI', 'DE'],
            'laggards': ['WV', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'KS', 'NE', 'SD', 'ND', 'MT', 'ID', 'WY', 'AK']
        }
        
        # Adoption multipliers based on time and state tier
        # Format: [early_adopters, early_majority, late_majority, laggards]
        adoption_timeline = {
            0: [0.7, 0.4, 0.25, 0.1],  # 2 years ago
            1: [0.9, 0.7, 0.5, 0.3],   # 1 year ago
            2: [1.0, 0.9, 0.8, 0.6]    # Current
        }
        
        # Generate data for each state, time period, and state
        for time_period in time_periods:
            year_delta = min(2, time_period['year'] - start_date.year)  # Cap at 2 years difference
            month = time_period['month']
            
            # Get seasonal and yearly adjustment factors
            season_factor = seasonal_factors[month]
            year_factor = yearly_growth.get(year_delta, yearly_growth[0])  # Use 0 if not found
            
            for region, states in regions.items():
                base_char = base_region_characteristics[region]
                
                # Adjust base characteristics for this time period
                char = {
                    'grid_range': (
                        base_char['grid_range'][0] * season_factor[0] * year_factor[0],
                        base_char['grid_range'][1] * season_factor[0] * year_factor[0]
                    ),
                    'solar_range': (
                        base_char['solar_range'][0] * season_factor[1] * year_factor[1],
                        base_char['solar_range'][1] * season_factor[1] * year_factor[1]
                    ),
                    'ev_range': (
                        base_char['ev_range'][0] * season_factor[2] * year_factor[2],
                        base_char['ev_range'][1] * season_factor[2] * year_factor[2]
                    ),
                    'ac_range': (
                        base_char['ac_range'][0] * season_factor[3] * year_factor[3],
                        base_char['ac_range'][1] * season_factor[3] * year_factor[3]
                    ),
                    'water_heater_range': (
                        base_char['water_heater_range'][0] * season_factor[4] * year_factor[4],
                        base_char['water_heater_range'][1] * season_factor[4] * year_factor[4]
                    ),
                    'pv_adoption_range': (
                        base_char['pv_adoption_range'][0] * year_factor[5],
                        base_char['pv_adoption_range'][1] * year_factor[5]
                    ),
                    'ev_adoption_range': (
                        base_char['ev_adoption_range'][0] * year_factor[6],
                        base_char['ev_adoption_range'][1] * year_factor[6]
                    )
                }
                
                for state in states:
                    # Determine which adoption tier this state belongs to
                    state_tier = None
                    for tier, states_list in state_adoption_tiers.items():
                        if state in states_list:
                            state_tier = tier
                            break
                    
                    # Get adoption multiplier based on state tier and time period
                    if state_tier == 'early_adopters':
                        tier_index = 0
                    elif state_tier == 'early_majority':
                        tier_index = 1
                    elif state_tier == 'late_majority':
                        tier_index = 2
                    else:  # laggards
                        tier_index = 3
                    
                    # Get adoption multiplier for this time period and state tier
                    adoption_multiplier = adoption_timeline[year_delta][tier_index]
                    
                    # Add some realistic variance within regions
                    variance_factor = random.uniform(0.8, 1.2)
                    
                    # Base metrics 
                    grid_consumption = random.uniform(*char['grid_range']) * variance_factor
                    
                    # Apply state-specific adoption multipliers
                    solar_production = random.uniform(*char['solar_range']) * variance_factor * adoption_multiplier
                    ev_charging = random.uniform(*char['ev_range']) * variance_factor * adoption_multiplier
                    ac_usage = random.uniform(*char['ac_range']) * variance_factor
                    water_heater = random.uniform(*char['water_heater_range']) * variance_factor
                    
                    # Adoption rates (percentages) with stronger historical progression
                    pv_adoption = random.uniform(*char['pv_adoption_range']) * adoption_multiplier
                    ev_adoption = random.uniform(*char['ev_adoption_range']) * adoption_multiplier
                    
                    # Energy efficiency improves over time, with early adopter states leading
                    efficiency_base = 55 + random.uniform(-5, 5)  # Base efficiency
                    efficiency_growth = year_factor[7] * adoption_multiplier  # Apply time and state tier factors
                    efficiency_score = min(98, efficiency_base * efficiency_growth)  # Cap at 98
                    
                    # Home counts for context - with growth over time
                    home_count = int(random.uniform(500, 5000) * (1 + year_delta * 0.05))
                    
                    # Calculate derived metrics
                    solar_coverage = (solar_production / grid_consumption * 100) if grid_consumption > 0 else 0

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
        .applymap(highlight_class_imbalance, subset=['Negative Class (%)', 'Positive Class (%)'])\
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
    
    # Load data
    states_data, households = generate_geo_data()
    us_states_geojson = load_us_geojson()
    
    # Create filter controls in sidebar
    st.sidebar.markdown("### Map Filters")
    show_ev = st.sidebar.checkbox("Show EV Chargers", value=True)
    show_ac = st.sidebar.checkbox("Show AC Units", value=True)
    show_pv = st.sidebar.checkbox("Show Solar Panels", value=True)
    
    # Get state from URL parameter if available
    params = st.query_params
    selected_state = params.get("state", [""])[0]
    
    if selected_state not in states_data:
        selected_state = ""
    
    # Filter households by selected state and device types
    filtered_households = [
        h for h in households if 
        (not selected_state or h['state'] == selected_state) and
        ((show_ev and h['has_ev']) or 
         (show_ac and h['has_ac']) or 
         (show_pv and h['has_pv']))
    ]
    
    # Create map
    if selected_state:
        state = states_data[selected_state]
        m = folium.Map(
            location=[state['lat'], state['lon']], 
            zoom_start=state['zoom'],
            tiles="CartoDB positron"
        )
    else:
        m = folium.Map(
            location=[39.8283, -98.5795],  # Center of US
            zoom_start=4,
            tiles="CartoDB positron"
        )
    
    # Add satellite view layer
    folium.TileLayer('Esri_WorldImagery', name='Satellite View', attr='Esri').add_to(m)
    
    if not selected_state and us_states_geojson:
        # Add state boundaries with click functionality
        style_function = lambda x: {
            'fillColor': primary_purple,
            'color': 'white',
            'weight': 1,
            'fillOpacity': 0.5
        }
        
        folium.GeoJson(
            us_states_geojson,
            style_function=style_function,
            highlight_function=lambda x: {
                'fillColor': light_purple,
                'color': 'white',
                'weight': 3,
                'fillOpacity': 0.7
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['name'],
                aliases=['State:'],
                labels=True,
                sticky=True
            ),
            popup=folium.GeoJsonPopup(
                fields=['name'],
                aliases=['Click to view devices in:']
            )
        ).add_to(m)
    
    # Create marker cluster for households
    marker_cluster = MarkerCluster().add_to(m)
    
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
            popup=popup_content,
            icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa'),
            tooltip=f"Home {house['id']}"
        ).add_to(marker_cluster)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; 
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid gray;">
        <h4>Legend</h4>
        <p><i class="fa fa-plug" style="color: blue;"></i> EV Charger</p>
        <p><i class="fa fa-snowflake-o" style="color: orange;"></i> AC Unit</p>
        <p><i class="fa fa-sun-o" style="color: green;"></i> Solar Panel</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Display the map
    folium_static(m, width=1000, height=600)
    
    # Display statistics if a state is selected
    if selected_state:
        state = states_data[selected_state]
        st.subheader(f"{state['name']} Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Homes", state['total_homes'])
        with col2:
            st.metric("EV Chargers", state['ev_homes'])
        with col3:
            st.metric("AC Units", state['ac_homes'])
        with col4:
            st.metric("Solar Panels", state['pv_homes'])
        
        # Display data table
        st.subheader("Homes Data")
        df = pd.DataFrame(filtered_households)
        st.dataframe(df)
