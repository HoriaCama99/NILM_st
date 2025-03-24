import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from PIL import Image
import seaborn as sns
import datetime

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
page = st.sidebar.radio("Select Page", ["Sample Output", "Performance Metrics"])

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
                    ev_percentage = (ev_charging / grid_consumption * 100) if grid_consumption > 0 else 0
                    ac_percentage = (ac_usage / grid_consumption * 100) if grid_consumption > 0 else 0
                    
                    geo_data.append({
                        'state': state,
                        'region': region,
                        'year': time_period['year'],
                        'month': time_period['month'],
                        'month_name': time_period['month_name'],
                        'quarter': time_period['quarter'],
                        'period_label': time_period['period_label'],
                        'date_value': time_period['date'],  # Store the actual date object for proper sorting
                        'grid_consumption': grid_consumption,
                        'solar_production': solar_production,
                        'ev_charging': ev_charging,
                        'ac_usage': ac_usage,
                        'water_heater_usage': water_heater,
                        'pv_adoption_rate': pv_adoption,
                        'ev_adoption_rate': ev_adoption,
                        'solar_coverage': solar_coverage,
                        'ev_percentage': ev_percentage,
                        'ac_percentage': ac_percentage,
                        'efficiency_score': efficiency_score,
                        'home_count': home_count
                    })
        
        return pd.DataFrame(geo_data)
    
    # Generate geographic data
    geo_df = generate_geo_data()
    
    # Add time period selection controls at the top
    st.subheader("Time Period Selection")
    
    # Get unique time periods
    time_periods = sorted(geo_df['period_label'].unique())
    
    # Create columns for different time controls
    time_control_col1, time_control_col2 = st.columns([3, 1])
    
    with time_control_col1:
        # Add time period slider - without using format_func which is not supported
        selected_time_index = st.slider(
            "Select Time Period",
            min_value=0,
            max_value=len(time_periods)-1,
            value=len(time_periods)-1  # Default to latest time period
        )
        # Display the currently selected time period
        selected_period = time_periods[selected_time_index]
        st.caption(f"Currently viewing: **{selected_period}**")
    
    with time_control_col2:
        # Empty column for balance
        st.markdown("")  # Just to keep the layout balanced
    
    # Filter by selected time period - only Single Period is now supported
    time_filtered_geo_df = geo_df[geo_df['period_label'] == selected_period]
    period_title = f"Data for {selected_period}"
    
    # Create a two-column layout for map controls and explanations
    st.subheader(f"Geographical Energy Insights: {period_title}")
    map_col1, map_col2 = st.columns([1, 3])
    
    with map_col1:
        # Add map control options
        map_metric = st.selectbox(
            "Select Map Metric",
            [
                "Grid Consumption (kWh)",
                "Solar Production (kWh)",
                "EV Charging (kWh)",
                "AC Usage (kWh)",
                "Solar Coverage (%)",
                "PV Adoption Rate (%)",
                "EV Adoption Rate (%)",
                "Energy Efficiency Score"
            ]
        )
        
        # Add region filter
        selected_regions = st.multiselect(
            "Filter by Region",
            ["Northeast", "Southeast", "Midwest", "Southwest", "West"],
            default=["Northeast", "Southeast", "Midwest", "Southwest", "West"]
        )
        
        # Filter data by selected regions and time
        if selected_regions:
            filtered_geo_df = time_filtered_geo_df[time_filtered_geo_df['region'].isin(selected_regions)]
        else:
            filtered_geo_df = time_filtered_geo_df
            
        # Add explanatory text about the selected metric
        metric_explanations = {
            "Grid Consumption (kWh)": "Average monthly electricity consumption from the grid per household.",
            "Solar Production (kWh)": "Average monthly solar energy production per household with PV systems.",
            "EV Charging (kWh)": "Average monthly electricity used for EV charging in homes with EVs.",
            "AC Usage (kWh)": "Average monthly electricity consumed by air conditioning systems.",
            "Solar Coverage (%)": "Percentage of grid consumption offset by solar production.",
            "PV Adoption Rate (%)": "Percentage of homes with solar PV systems installed.",
            "EV Adoption Rate (%)": "Percentage of homes with electric vehicles.",
            "Energy Efficiency Score": "Overall energy efficiency score (higher is better)."
        }
        
        st.markdown(f"""
        ### About This Metric
        
        **{map_metric}**
        
        {metric_explanations.get(map_metric, "")}
        
        This map shows variations across different states and regions, highlighting geographic patterns in energy usage and technology adoption.
        """)
    
    with map_col2:
        # Add state click handling using session state
        if 'selected_state' not in st.session_state:
            st.session_state.selected_state = None
            st.session_state.zoomed_in = False
        
        # Set up the color scales for different metrics
        color_scales = {
            "Grid Consumption (kWh)": [light_purple, primary_purple, dark_purple],
            "Solar Production (kWh)": ["#F9F0D9", cream, green],
            "EV Charging (kWh)": ["#D9DCFF", light_purple, primary_purple],
            "AC Usage (kWh)": ["#F9F0D9", green, "#43867F"],
            "Solar Coverage (%)": ["#F9F0D9", cream, green],
            "PV Adoption Rate (%)": ["#F9F0D9", cream, green],
            "EV Adoption Rate (%)": ["#D9DCFF", light_purple, primary_purple],
            "Energy Efficiency Score": [salmon, cream, green]
        }
        
        # Map metric name to dataframe column
        metric_mappings = {
            "Grid Consumption (kWh)": "grid_consumption",
            "Solar Production (kWh)": "solar_production",
            "EV Charging (kWh)": "ev_charging",
            "AC Usage (kWh)": "ac_usage",
            "Solar Coverage (%)": "solar_coverage",
            "PV Adoption Rate (%)": "pv_adoption_rate",
            "EV Adoption Rate (%)": "ev_adoption_rate",
            "Energy Efficiency Score": "efficiency_score"
        }
        
        selected_metric = metric_mappings[map_metric]
        
        # Generate mock home data for the selected state when zoomed in
        @st.cache_data
        def generate_home_data(state, count=50):
            import random
            from geopy.geocoders import Nominatim
            
            # State bounding boxes (approximations)
            state_bounds = {
                'AL': (30.1, -88.5, 35.0, -84.9),  # min_lat, min_lon, max_lat, max_lon
                'AK': (51.0, -179.0, 71.5, -130.0),
                'AZ': (31.3, -114.8, 37.0, -109.0),
                'AR': (33.0, -94.6, 36.5, -89.6),
                'CA': (32.5, -124.4, 42.0, -114.1),
                'CO': (37.0, -109.1, 41.0, -102.0),
                'CT': (40.9, -73.7, 42.1, -71.8),
                'DE': (38.4, -75.8, 39.9, -75.0),
                'FL': (24.5, -87.6, 31.0, -80.0),
                'GA': (30.5, -85.6, 35.0, -80.8),
                'HI': (18.7, -160.3, 22.3, -154.8),
                'ID': (42.0, -117.2, 49.0, -111.0),
                'IL': (36.9, -91.5, 42.5, -87.0),
                'IN': (37.8, -88.1, 41.8, -84.8),
                'IA': (40.3, -96.6, 43.5, -90.1),
                'KS': (37.0, -102.1, 40.0, -94.6),
                'KY': (36.5, -89.6, 39.2, -81.9),
                'LA': (29.0, -94.1, 33.0, -89.0),
                'ME': (43.0, -71.1, 47.5, -66.9),
                'MD': (38.0, -79.5, 39.7, -75.0),
                'MA': (41.2, -73.5, 42.9, -69.9),
                'MI': (41.7, -90.4, 48.3, -82.1),
                'MN': (43.5, -97.2, 49.4, -89.5),
                'MS': (30.1, -91.7, 35.0, -88.1),
                'MO': (36.0, -95.8, 40.6, -89.1),
                'MT': (44.3, -116.1, 49.0, -104.0),
                'NE': (40.0, -104.1, 43.0, -95.3),
                'NV': (35.0, -120.0, 42.0, -114.0),
                'NH': (42.7, -72.6, 45.3, -70.6),
                'NJ': (38.9, -75.6, 41.4, -73.9),
                'NM': (31.3, -109.1, 37.0, -103.0),
                'NY': (40.5, -79.8, 45.0, -71.8),
                'NC': (33.8, -84.3, 36.6, -75.5),
                'ND': (45.9, -104.1, 49.0, -96.6),
                'OH': (38.4, -84.8, 42.0, -80.5),
                'OK': (33.6, -103.0, 37.0, -94.4),
                'OR': (42.0, -124.6, 46.3, -116.5),
                'PA': (39.7, -80.5, 42.3, -74.7),
                'RI': (41.1, -71.9, 42.0, -71.1),
                'SC': (32.0, -83.4, 35.2, -78.5),
                'SD': (42.5, -104.1, 46.0, -96.4),
                'TN': (34.9, -90.3, 36.7, -81.6),
                'TX': (25.8, -106.7, 36.5, -93.5),
                'UT': (37.0, -114.1, 42.0, -109.0),
                'VT': (42.7, -73.5, 45.0, -71.5),
                'VA': (36.5, -83.7, 39.5, -75.2),
                'WA': (45.5, -124.8, 49.0, -116.9),
                'WV': (37.2, -82.7, 40.6, -77.7),
                'WI': (42.5, -92.9, 47.1, -86.8),
                'WY': (41.0, -111.1, 45.0, -104.0)
            }
            
            # Use the bounding box for the state or default to a generic one
            if state in state_bounds:
                min_lat, min_lon, max_lat, max_lon = state_bounds[state]
            else:
                # Default bounding box if state is not found
                min_lat, min_lon, max_lat, max_lon = 35.0, -120.0, 45.0, -100.0
            
            # Generate random home locations within the state bounds
            homes = []
            for i in range(count):
                lat = min_lat + (max_lat - min_lat) * random.random()
                lon = min_lon + (max_lon - min_lon) * random.random()
                
                # Determine device presence
                has_ev = random.random() < 0.4  # 40% chance of having an EV
                has_pv = random.random() < 0.5  # 50% chance of having PV
                has_ac = random.random() < 0.75  # 75% chance of having AC
                
                # City names - some major cities in each state or generic names
                cities = {
                    'CA': ['Los Angeles', 'San Francisco', 'San Diego', 'Sacramento', 'Fresno'],
                    'NY': ['New York', 'Buffalo', 'Albany', 'Syracuse', 'Rochester'],
                    'TX': ['Houston', 'Dallas', 'Austin', 'San Antonio', 'El Paso'],
                    # Add more as needed
                }
                
                if state in cities:
                    city = random.choice(cities[state])
                else:
                    generic_cities = ['Downtown', 'Uptown', 'Midtown', 'Westside', 'Eastside', 'Northend', 'Southside']
                    city = f"{random.choice(generic_cities)}, {state}"
                
                # Energy consumption values
                grid = random.uniform(500, 1500)
                solar = random.uniform(100, 800) if has_pv else 0
                ev_charging = random.uniform(100, 400) if has_ev else 0
                ac_usage = random.uniform(100, 600) if has_ac else 0
                
                homes.append({
                    'id': f"home_{i}",
                    'lat': lat,
                    'lon': lon,
                    'city': city,
                    'has_ev': has_ev,
                    'has_pv': has_pv,
                    'has_ac': has_ac,
                    'grid': grid,
                    'solar': solar,
                    'ev_charging': ev_charging,
                    'ac_usage': ac_usage
                })
            
            return pd.DataFrame(homes)

        # Create the map visualization with click events
        fig = px.choropleth(
            filtered_geo_df,
            locations='state',
            color=selected_metric,
            locationmode="USA-states",
            scope="usa",
            color_continuous_scale=color_scales[map_metric],
            range_color=[filtered_geo_df[selected_metric].min(), filtered_geo_df[selected_metric].max()],
            hover_name='state',
            hover_data={
                'state': False,
                'region': True,
                'grid_consumption': ':.1f',
                'solar_production': ':.1f',
                'ev_charging': ':.1f',
                'ac_usage': ':.1f',
                'pv_adoption_rate': ':.1f',
                'ev_adoption_rate': ':.1f',
                'solar_coverage': ':.1f',
                'home_count': True
            },
            labels={
                'grid_consumption': 'Grid (kWh)',
                'solar_production': 'Solar (kWh)',
                'ev_charging': 'EV (kWh)',
                'ac_usage': 'AC (kWh)',
                'pv_adoption_rate': 'PV Adoption (%)',
                'ev_adoption_rate': 'EV Adoption (%)',
                'solar_coverage': 'Solar Coverage (%)',
                'home_count': 'Homes'
            }
        )
        
        # Update map layout with clickable states
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor=white,
            geo=dict(
                showlakes=True,
                lakecolor=white,
                showsubunits=True,
                subunitcolor="lightgray"
            ),
            coloraxis_colorbar=dict(
                title=dict(
                    text=map_metric,
                    font=dict(color=dark_purple)
                ),
                tickfont=dict(color=dark_purple)
            ),
            height=550
        )
        
        # Display the map with click events
        map_chart = st.plotly_chart(fig, use_container_width=True)

        # Add a container for click events 
        click_container = st.container()

        # Capture clicks on the map - this requires JavaScript callbacks
        clicked_state = None

        # Add a small note above the map
        st.markdown("👆 **Click on any state in the map to see its trend over time**")
        
        # Add a checkbox to toggle the state trend section (simulating the functionality of clicking a state)
        show_trend_analysis = st.checkbox("Show State Trend Analysis", value=False)
        
        # Add state selection dropdown
        selected_state = None
        state_selector = st.selectbox(
            "Select a state to see home details:",
            [""] + sorted(filtered_geo_df['state'].unique()),
            key="state_selector"
        )
        
        if state_selector:
            selected_state = state_selector
            st.session_state.selected_state = selected_state
            st.session_state.zoomed_in = True
        
        # Check if we're in zoomed state view
        if st.session_state.zoomed_in and st.session_state.selected_state:
            # Create a container for the state zoom view
            zoom_container = st.container()
            
            with zoom_container:
                if st.button("← Back to National Map", key="back_to_map"):
                    st.session_state.zoomed_in = False
                    st.session_state.selected_state = None
                    st.experimental_rerun()
                
                st.subheader(f"Home Details in {st.session_state.selected_state}")
                
                # Generate the home data for this state
                home_data = generate_home_data(st.session_state.selected_state)
                
                # Add device filter controls
                filter_cols = st.columns(3)
                with filter_cols[0]:
                    show_ev = st.checkbox("Show EV Homes", value=True)
                with filter_cols[1]:
                    show_pv = st.checkbox("Show Solar PV Homes", value=True)
                with filter_cols[2]:
                    show_ac = st.checkbox("Show AC Homes", value=True)
                
                # Apply filters
                filtered_homes = home_data.copy()
                if show_ev:
                    filtered_homes = filtered_homes[filtered_homes['has_ev']]
                if show_pv:
                    filtered_homes = filtered_homes[filtered_homes['has_pv']]
                if show_ac:
                    filtered_homes = filtered_homes[filtered_homes['has_ac']]
                
                # Show count of filtered homes
                if len(filtered_homes) > 0:
                    st.markdown(f"Showing **{len(filtered_homes)}** homes with selected devices")
                else:
                    st.warning("No homes match your filter criteria. Try adjusting your filters.")
                
                # Create the scatter map for homes
                if len(filtered_homes) > 0:
                    fig = px.scatter_mapbox(
                        filtered_homes,
                        lat="lat",
                        lon="lon",
                        hover_name="city",
                        hover_data={
                            "has_ev": True,
                            "has_pv": True,
                            "has_ac": True,
                            "grid": ":.1f",
                            "solar": ":.1f",
                            "ev_charging": ":.1f",
                            "ac_usage": ":.1f",
                            "lat": False,
                            "lon": False
                        },
                        color_discrete_sequence=[primary_purple],
                        zoom=6,
                        mapbox_style="carto-positron"
                    )
                    
                    # Add different markers based on what devices the home has
                    marker_sizes = filtered_homes.apply(
                        lambda row: 10 + (5 if row['has_ev'] else 0) + (5 if row['has_pv'] else 0) + (5 if row['has_ac'] else 0),
                        axis=1
                    )
                    
                    # Add customization to the pins
                    fig.update_traces(
                        marker=dict(
                            size=marker_sizes, 
                            opacity=0.7,
                            color=filtered_homes.apply(
                                lambda row: primary_purple if row['has_ev'] else (green if row['has_pv'] else salmon),
                                axis=1
                            )
                        ),
                        selector=dict(mode='markers')
                    )
                    
                    # Improve hover template
                    fig.update_traces(
                        hovertemplate="<b>%{hovertext}</b><br>" +
                                    "EV Charging: %{customdata[0]}<br>" +
                                    "Solar PV: %{customdata[1]}<br>" +
                                    "AC: %{customdata[2]}<br>" +
                                    "Grid: %{customdata[3]} kWh<br>" +
                                    "Solar: %{customdata[4]} kWh<br>" +
                                    "EV Charging: %{customdata[5]} kWh<br>" +
                                    "AC Usage: %{customdata[6]} kWh<br>"
                    )
                    
                    # Update layout
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=550,
                        mapbox=dict(
                            center=dict(
                                lat=filtered_homes['lat'].mean(),
                                lon=filtered_homes['lon'].mean()
                            )
                        )
                    )
                    
                    # Display the map
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display summary statistics
                    stats_cols = st.columns(3)
                    with stats_cols[0]:
                        st.metric(
                            "Average Grid Consumption",
                            f"{filtered_homes['grid'].mean():.1f} kWh"
                        )
                    
                    with stats_cols[1]:
                        if show_pv and len(filtered_homes[filtered_homes['has_pv']]) > 0:
                            st.metric(
                                "Average Solar Production",
                                f"{filtered_homes[filtered_homes['has_pv']]['solar'].mean():.1f} kWh"
                            )
                        else:
                            st.metric("Average Solar Production", "N/A")
                    
                    with stats_cols[2]:
                        if show_ev and len(filtered_homes[filtered_homes['has_ev']]) > 0:
                            st.metric(
                                "Average EV Charging",
                                f"{filtered_homes[filtered_homes['has_ev']]['ev_charging'].mean():.1f} kWh"
                            )
                        else:
                            st.metric("Average EV Charging", "N/A")

        # Show trend analysis if selected and not zoomed into a state map
        if show_trend_analysis and not st.session_state.zoomed_in:
            # State dropdown for selecting a state to view trends
            st.subheader("State Trend Analysis")
            # ... rest of the trend analysis code ...

# The rest of your code continues normally after this section
else:  # Performance Metrics page
    # Define the load_performance_data function properly
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
            }
        }
        
        return models_data

    # Load performance data
    models_data = load_performance_data()
    
    # Continue with the rest of the code...
