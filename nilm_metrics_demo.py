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
        # Set up the color scales for different metrics
        color_scales = {
            "Grid Consumption (kWh)": [light_purple, primary_purple, dark_purple],
            "Solar Production (kWh)": ["#F9F0D9", cream, green],
            "EV Charging (kWh)": ["#D9DCFF", light_purple, primary_purple],
            "AC Usage (kWh)": ["#D9F2EC", green, "#43867F"],
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
        
        # Use session state to track which state is selected
        if 'selected_state' not in st.session_state:
            st.session_state.selected_state = None
            
        if 'show_zoomed_state' not in st.session_state:
            st.session_state.show_zoomed_state = False
        
        # Add custom JavaScript for capturing clicks
        js_code = """
        <script>
            const figure = document.querySelector('div[data-testid="stPlotlyChart"] .js-plotly-plot');
            if (figure) {
                figure.on('plotly_click', function(data) {
                    const clickedState = data.points[0].customdata;
                    if (clickedState) {
                        // Send message to Streamlit
                        const message = {
                            type: "streamlit:setComponentValue",
                            value: clickedState,
                            dataType: "str"
                        };
                        window.parent.postMessage(message, "*");
                        
                        // Auto-submit the form to trigger a rerun
                        setTimeout(function() {
                            window.parent.document.querySelector('button[kind="secondaryFormSubmit"]').click();
                        }, 100);
                    }
                });
            }
        </script>
        """
        
        # State coordinates for centering the map on selected state
        state_centers = {
            'AL': (32.7794, -86.8287), 'AK': (64.0685, -152.2782), 'AZ': (34.2744, -111.6602),
            'AR': (34.8938, -92.4426), 'CA': (37.1841, -119.4696), 'CO': (38.9972, -105.5478),
            'CT': (41.6219, -72.7273), 'DE': (38.9896, -75.5050), 'FL': (28.6305, -82.4497),
            'GA': (32.6415, -83.4426), 'HI': (20.2927, -156.3737), 'ID': (44.3509, -114.6130),
            'IL': (40.0417, -89.1965), 'IN': (39.8942, -86.2816), 'IA': (42.0751, -93.4960),
            'KS': (38.4937, -98.3804), 'KY': (37.5347, -85.3021), 'LA': (31.0689, -91.9968),
            'ME': (45.3695, -69.2428), 'MD': (39.0550, -76.7909), 'MA': (42.2596, -71.8083),
            'MI': (44.3467, -85.4102), 'MN': (46.2807, -94.3053), 'MS': (32.7364, -89.6678),
            'MO': (38.3566, -92.4580), 'MT': (47.0527, -109.6333), 'NE': (41.5378, -99.7951),
            'NV': (39.3289, -116.6312), 'NH': (43.6805, -71.5811), 'NJ': (40.1907, -74.6728),
            'NM': (34.4071, -106.1126), 'NY': (42.9538, -75.5268), 'NC': (35.5557, -79.3877),
            'ND': (47.4501, -100.4659), 'OH': (40.2862, -82.7937), 'OK': (35.5889, -97.4943),
            'OR': (43.9336, -120.5583), 'PA': (40.8781, -77.7996), 'RI': (41.6762, -71.5562),
            'SC': (33.9169, -80.8964), 'SD': (44.4443, -100.2263), 'TN': (35.8600, -86.3505),
            'TX': (31.4757, -99.3312), 'UT': (39.3055, -111.6703), 'VT': (44.0687, -72.6658),
            'VA': (37.5215, -78.8537), 'WA': (47.3826, -120.4472), 'WV': (38.6409, -80.6227),
            'WI': (44.6243, -89.9941), 'WY': (42.9957, -107.5512)
        }
        
        # Add buttons above the map for interaction controls
        col1, col2, col3 = st.columns([2, 3, 2])
        with col1:
            # Dropdown for state selection (alternative to clicking)
            selected_state = st.selectbox(
                "Select a state:",
                options=[''] + sorted(filtered_geo_df['state'].unique()),
                key="state_selector",
                index=0
            )
        
        with col2:
            st.markdown("👆 **Click on any state in the map to zoom in and view household data**")
        
        with col3:
            # Reset button to return to national view
            reset_view = st.button("Reset Map View")
        
        # Process state selection (either from click or dropdown)
        if selected_state or ('selected_state' in st.session_state and st.session_state.selected_state):
            # Update session state
            if selected_state and selected_state != st.session_state.selected_state:
                st.session_state.selected_state = selected_state
                st.session_state.show_zoomed_state = True
        
        # Reset map view if requested
        if reset_view:
            st.session_state.selected_state = None
            st.session_state.show_zoomed_state = False
            st.rerun()
        
        # Get the current state selection
        zoom_state = None
        if 'selected_state' in st.session_state and st.session_state.selected_state:
            zoom_state = st.session_state.selected_state
        
        # Function to generate random household data
        def generate_households(state_code, count=100):
            # Get state data from our dataset
            state_info = filtered_geo_df[filtered_geo_df['state'] == state_code].iloc[0]
            
            # Get state center coordinates
            if state_code not in state_centers:
                # Default to middle of US if state not found
                center_lat, center_lon = (39.8283, -98.5795)
            else:
                center_lat, center_lon = state_centers[state_code]
            
            # Generate random households around the state center
            import random
            households = []
            
            # Use the state's adoption rates to determine device presence probabilities
            ev_prob = state_info['ev_adoption_rate'] / 100
            pv_prob = state_info['pv_adoption_rate'] / 100
            ac_prob = 0.7  # AC is common in most homes
            
            for i in range(count):
                # Generate a random point within the state (within ~50 miles of center)
                lat = center_lat + (random.random() - 0.5) * 0.7
                lon = center_lon + (random.random() - 0.5) * 0.7
                
                # Determine device presence
                has_ev = random.random() < ev_prob
                has_pv = random.random() < pv_prob
                has_ac = random.random() < ac_prob
                
                # Generate household energy data based on state averages and device presence
                grid_consumption = state_info['grid_consumption'] * (0.7 + random.random() * 0.6)
                solar_production = state_info['solar_production'] * (0.7 + random.random() * 0.6) if has_pv else 0
                ev_charging = state_info['ev_charging'] * (0.7 + random.random() * 0.6) if has_ev else 0
                ac_usage = state_info['ac_usage'] * (0.7 + random.random() * 0.6) if has_ac else 0
                
                households.append({
                    'lat': lat,
                    'lon': lon,
                    'has_ev': has_ev,
                    'has_pv': has_pv,
                    'has_ac': has_ac,
                    'grid_consumption': grid_consumption,
                    'solar_production': solar_production,
                    'ev_charging': ev_charging,
                    'ac_usage': ac_usage,
                    'household_id': f"H{i+1}"
                })
            
            return pd.DataFrame(households)
        
        # Create the main visualization (choropleth for states)
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
        
        # Update map layout based on whether zoomed in or not
        if zoom_state and 'show_zoomed_state' in st.session_state and st.session_state.show_zoomed_state:
            # Generate household data for selected state
            households_df = generate_households(zoom_state, count=100)
            
            # Extract state center coordinates
            if zoom_state in state_centers:
                center_lat, center_lon = state_centers[zoom_state]
            else:
                center_lat, center_lon = (39.8283, -98.5795)  # Default to US center
            
            # Add device filter controls if zoomed in
            filter_cols = st.columns(4)
            with filter_cols[0]:
                show_ev = st.checkbox("Show EV Homes", value=True)
            with filter_cols[1]:
                show_pv = st.checkbox("Show PV Homes", value=True)
            with filter_cols[2]:
                show_ac = st.checkbox("Show AC Homes", value=True)
            with filter_cols[3]:
                show_none = st.checkbox("Show Homes without devices", value=False)
            
            # Apply filters to household data
            filtered_households = households_df.copy()
            device_filters = []
            
            if show_ev:
                device_filters.append(filtered_households['has_ev'] == True)
            if show_pv:
                device_filters.append(filtered_households['has_pv'] == True)
            if show_ac:
                device_filters.append(filtered_households['has_ac'] == True)
            
            if device_filters:
                # Combine with OR if any device filter is active
                combined_filter = device_filters[0]
                for f in device_filters[1:]:
                    combined_filter = combined_filter | f
                
                # Apply the filter
                if not show_none:
                    filtered_households = filtered_households[combined_filter]
            elif not show_none:
                # If no device filters are active but show_none is False, show no households
                filtered_households = filtered_households[filtered_households['household_id'] == "NONE"]
            
            # Create device-specific dataframes for layered visualization
            ev_households = filtered_households[filtered_households['has_ev']]
            pv_households = filtered_households[filtered_households['has_pv']]
            ac_households = filtered_households[filtered_households['has_ac']]
            basic_households = filtered_households[~(filtered_households['has_ev'] | filtered_households['has_pv'] | filtered_households['has_ac'])]
            
            # Add households with no special devices
            if len(basic_households) > 0 and show_none:
                fig.add_trace(go.Scattergeo(
                    lon=basic_households['lon'],
                    lat=basic_households['lat'],
                    text=basic_households['household_id'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=light_purple,
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    name='Basic Homes',
                    hovertemplate='<b>%{text}</b><br>No special devices<br><extra></extra>'
                ))
            
            # Add AC households
            if len(ac_households) > 0 and show_ac:
                fig.add_trace(go.Scattergeo(
                    lon=ac_households['lon'],
                    lat=ac_households['lat'],
                    text=ac_households['household_id'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=green,
                        opacity=0.8,
                        line=dict(width=1, color='white'),
                        symbol='circle'
                    ),
                    name='AC Homes',
                    hovertemplate='<b>%{text}</b><br>Has AC<br>AC: %{customdata[0]:.1f} kWh<br>Grid: %{customdata[1]:.1f} kWh<br><extra></extra>',
                    customdata=ac_households[['ac_usage', 'grid_consumption']]
                ))
            
            # Add PV households
            if len(pv_households) > 0 and show_pv:
                fig.add_trace(go.Scattergeo(
                    lon=pv_households['lon'],
                    lat=pv_households['lat'],
                    text=pv_households['household_id'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=cream,
                        opacity=0.8,
                        line=dict(width=1, color='white'),
                        symbol='diamond'
                    ),
                    name='PV Homes',
                    hovertemplate='<b>%{text}</b><br>Has Solar PV<br>Solar: %{customdata[0]:.1f} kWh<br>Grid: %{customdata[1]:.1f} kWh<br><extra></extra>',
                    customdata=pv_households[['solar_production', 'grid_consumption']]
                ))
            
            # Add EV households
            if len(ev_households) > 0 and show_ev:
                fig.add_trace(go.Scattergeo(
                    lon=ev_households['lon'],
                    lat=ev_households['lat'],
                    text=ev_households['household_id'],
                    mode='markers',
                    marker=dict(
                        size=14,
                        color=primary_purple,
                        opacity=0.9,
                        line=dict(width=1, color='white'),
                        symbol='star'
                    ),
                    name='EV Homes',
                    hovertemplate='<b>%{text}</b><br>Has EV Charging<br>EV: %{customdata[0]:.1f} kWh<br>Grid: %{customdata[1]:.1f} kWh<br><extra></extra>',
                    customdata=ev_households[['ev_charging', 'grid_consumption']]
                ))
            
            # Update map layout for zoomed view
            fig.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor=white,
                geo=dict(
                    scope='usa',
                    center=dict(lat=center_lat, lon=center_lon),
                    projection_scale=6,  # Higher value = more zoomed in
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
                title=f"Household Energy Data in {zoom_state}",
                legend=dict(
                    orientation='h',
                    y=1.02,
                    x=0.5,
                    xanchor='center',
                    bgcolor=white,
                    font=dict(color=dark_purple)
                ),
                height=550
            )
            
            # Display household statistics for the zoomed state
            # These appear below the map
            st.subheader(f"Household Statistics for {zoom_state}")
            
            stat_cols = st.columns(4)
            with stat_cols[0]:
                st.metric(
                    label="Total Households",
                    value=len(filtered_households),
                    help="Number of households shown with current filters"
                )
            
            with stat_cols[1]:
                ev_percent = (len(ev_households) / len(filtered_households) * 100) if len(filtered_households) > 0 else 0
                st.metric(
                    label="EV Adoption",
                    value=f"{ev_percent:.1f}%",
                    help="Percentage of households with EV charging"
                )
            
            with stat_cols[2]:
                pv_percent = (len(pv_households) / len(filtered_households) * 100) if len(filtered_households) > 0 else 0
                st.metric(
                    label="Solar PV Adoption",
                    value=f"{pv_percent:.1f}%",
                    help="Percentage of households with solar PV systems"
                )
            
            with stat_cols[3]:
                ac_percent = (len(ac_households) / len(filtered_households) * 100) if len(filtered_households) > 0 else 0
                st.metric(
                    label="AC Usage",
                    value=f"{ac_percent:.1f}%",
                    help="Percentage of households with air conditioning"
                )
        else:
            # Standard national map view
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
        
        # Add click event handling
        fig.update_traces(
            customdata=filtered_geo_df['state'],
            hovertemplate='<b>%{hovertext}</b><br><extra></extra>'
        )
        
        # Display the map with click events
        map_container = st.container()
        with map_container:
            map_chart = st.plotly_chart(fig, use_container_width=True)
            st.markdown(js_code, unsafe_allow_html=True)
            
        # Note about the data if zoomed in
        if zoom_state and 'show_zoomed_state' in st.session_state and st.session_state.show_zoomed_state:
            st.caption("Note: Household data is generated based on state-level metrics and is for demonstration purposes.")

        # After the map visualization, add key statistics
        # Add summary statistics for the selected metric
        metric_col = map_metric.split(" (")[0].lower().replace(" ", "_")
        if metric_col in filtered_geo_df.columns:
            avg_value = filtered_geo_df[metric_col].mean()
            min_value = filtered_geo_df[metric_col].min()
            max_value = filtered_geo_df[metric_col].max()
            range_value = max_value - min_value
            
            st.markdown("### Key Statistics")
            
            # Create inline metrics with the same styling as Key Metrics section - full width
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric(
                    label="Average",
                    value=f"{avg_value:.1f}",
                    help=f"Average {map_metric.lower()} across selected regions"
                )
                st.markdown("</div>", unsafe_allow_html=True)
                
            with stat_col2:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric(
                    label="Minimum",
                    value=f"{min_value:.1f}",
                    help=f"Lowest {map_metric.lower()} in selected regions"
                )
                st.markdown("</div>", unsafe_allow_html=True)
                
            with stat_col3:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric(
                    label="Maximum",
                    value=f"{max_value:.1f}",
                    help=f"Highest {map_metric.lower()} in selected regions"
                )
                st.markdown("</div>", unsafe_allow_html=True)
                
            with stat_col4:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric(
                    label="Range",
                    value=f"{range_value:.1f}",
                    help=f"Difference between highest and lowest values"
                )
                st.markdown("</div>", unsafe_allow_html=True)

    # Solar Production Coverage Analysis 
    st.subheader("Solar Production Coverage Analysis")

    # Check if we have any homes with solar in the filtered dataset
    if filtered_df['pv detected'].sum() == 0:
        st.info("No homes with solar production in the current filtered dataset.")
    else:
        # Solar homes only
        solar_homes = filtered_df[filtered_df['pv detected'] == 1].copy()
        
        # Use actual grid consumption for coverage calculation
        solar_homes['solar_coverage'] = 100 * solar_homes.apply(
            lambda row: row['solar production (kWh)'] / row['grid (kWh)'] 
            if row['grid (kWh)'] > 0 else 0, 
            axis=1
        )
        
        # Create coverage categories for pie chart
        def categorize_coverage(coverage):
            if coverage >= 100:
                return "Exceeds Consumption (100%+)"
            elif coverage >= 75:
                return "High Coverage (75-99%)"
            elif coverage >= 50:
                return "Medium Coverage (50-74%)"
            elif coverage >= 25:
                return "Low Coverage (25-49%)"
            else:
                return "Minimal Coverage (<25%)"
        
        solar_homes['coverage_category'] = solar_homes['solar_coverage'].apply(categorize_coverage)
        
        # Count homes in each category
        category_counts = solar_homes['coverage_category'].value_counts().reset_index()
        category_counts.columns = ['Coverage Category', 'Number of Homes']
        
        # Define colors and order for pie chart
        category_order = [
            "Exceeds Consumption (100%+)",
            "High Coverage (75-99%)",
            "Medium Coverage (50-74%)",
            "Low Coverage (25-49%)",
            "Minimal Coverage (<25%)"
        ]
        
        # Keep only categories that exist in the data
        category_order = [cat for cat in category_order if cat in category_counts['Coverage Category'].values]
        
        # Sort the dataframe by our custom order
        category_counts['order'] = category_counts['Coverage Category'].apply(lambda x: category_order.index(x) if x in category_order else 999)
        category_counts = category_counts.sort_values('order').drop('order', axis=1)
        
        # Simple metrics row with team-colored containers
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            # Count homes where solar production exceeds consumption
            net_positive = (solar_homes['solar_coverage'] >= 100).sum()
            avg_coverage = solar_homes['solar_coverage'].mean()
            
            st.metric(
                label="Solar Homes",
                value=f"{len(solar_homes)} ({net_positive} net positive)",
                help="Number of homes with solar production detected"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric(
                label="Average Solar Coverage",
                value=f"{avg_coverage:.1f}%",
                help="Average percentage of grid consumption covered by solar production"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Create pie chart with team colors
        fig = px.pie(
            category_counts,
            values='Number of Homes',
            names='Coverage Category',
            title='Distribution of Solar Coverage Levels',
            color='Coverage Category',
            color_discrete_map={
                "Exceeds Consumption (100%+)": green,
                "High Coverage (75-99%)": cream,
                "Medium Coverage (50-74%)": primary_purple,
                "Low Coverage (25-49%)": dark_purple,
                "Minimal Coverage (<25%)": salmon
            },
        )
        
        # Update layout 
        fig.update_layout(
            legend_title="Coverage Level",
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor=white,
            plot_bgcolor=white,
            font=dict(color=dark_purple),
            title_font=dict(color=primary_purple),
            legend=dict(font=dict(color=dark_purple))
        )
        
        fig.update_traces(
            textinfo='percent+label',
            hovertemplate='%{label}<br>%{value} homes<br>%{percent}',
            textfont=dict(color=white)  
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Minimal explanation 
        st.markdown(f"""
        <div style="color:{primary_purple}; font-style:italic; text-align:center; margin-top:-20px;">
            Solar coverage shows what percentage of each home's total grid consumption is offset by solar production.
        </div>
        """, unsafe_allow_html=True)

    # Add footer with primary purple color
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; color:{primary_purple}; padding: 10px; border-radius: 5px;">
        This sample output demonstrates the type of insights available from the disaggregation model. 
        In a full deployment, thousands of households would be analyzed to provide statistically significant patterns and trends.
    </div>
    """, unsafe_allow_html=True)

else:  # Performance Metrics page
    # Just one title, with a more comprehensive subheader
    st.title("NILM Algorithm Performance Dashboard")
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
            }
        }
        
        return models_data

    # Load performance data
    models_data = load_performance_data()

    # Model selection in sidebar
    selected_model = st.sidebar.selectbox(
        "Select Model",
        ["V1", "V2", "V3", "V4", "V5"],
        index=4,  # Default to V5
        format_func=lambda x: x  # No need to transform the names anymore
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
    This dashboard presents performance metrics for the **{selected_model}** model 
    in detecting EV Charging, AC Usage, PV Usage, and WH Usage consumption patterns. For this benchmark, we have used four versions of the model (V1-V4), each with different hyperparameters and training strategies.
    """)

    # Main content area for Performance Metrics page
    st.subheader(f"Model: {selected_model}")

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
        
        for model in ["V1", "V2", "V3", "V4", "V5"]:
            trend_data.append({
                "Model": model,
                "DPSPerc (%)": models_data[model]["DPSPerc"][trend_device],
                "FPR (%)": models_data[model]["FPR"][trend_device] * 100,  # Convert to percentage
                "TECA (%)": models_data[model]["TECA"][trend_device] * 100  # Convert to percentage
            })
        
        trend_df = pd.DataFrame(trend_data)
        
        # Find best model for each metric
        best_dpsperc_model = trend_df.loc[trend_df["DPSPerc (%)"].idxmax()]["Model"]
        best_fpr_model = trend_df.loc[trend_df["FPR (%)"].idxmin()]["Model"]  # Lowest is best for FPR
        best_teca_model = trend_df.loc[trend_df["TECA (%)"].idxmax()]["Model"]
        
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
        dpsperc_best_idx = trend_df[trend_df["Model"] == best_dpsperc_model].index[0]
        fig.add_trace(go.Scatter(
            x=[best_dpsperc_model],
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
        fpr_best_idx = trend_df[trend_df["Model"] == best_fpr_model].index[0]
        fig.add_trace(go.Scatter(
            x=[best_fpr_model],
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
        teca_best_idx = trend_df[trend_df["Model"] == best_teca_model].index[0]
        fig.add_trace(go.Scatter(
            x=[best_teca_model],
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
        
        # Highlight the currently selected model with circles
        current_model_index = ["V1", "V2", "V3", "V4", "V5"].index(selected_model)
        
        # Add vertical line for currently selected model
        fig.add_shape(
            type="line",
            x0=selected_model,
            y0=0,
            x1=selected_model,
            y1=100,
            line=dict(
                color="rgba(80, 80, 80, 0.3)",
                width=6,
                dash="dot",
            )
        )
        
        # Add timeline elements - faint vertical lines and model version labels at bottom
        for model in ["V1", "V2", "V3", "V4", "V5"]:
            if model != selected_model:  # We already added a line for the selected model
                fig.add_shape(
                    type="line",
                    x0=model,
                    y0=0,
                    x1=model,
                    y1=100,
                    line=dict(
                        color="rgba(200, 200, 200, 0.3)",
                        width=1,
                    )
                )
        
        # Determine which model has best average performance
        trend_df['FPR_inv'] = 100 - trend_df['FPR (%)']  # Invert FPR so higher is better
        trend_df['avg_score'] = (trend_df['DPSPerc (%)'] + trend_df['FPR_inv'] + trend_df['TECA (%)']) / 3
        best_overall = trend_df.loc[trend_df['avg_score'].idxmax()]['Model']

        # Add highlight rectangle for the best overall model
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=float(["V1", "V2", "V3", "V4", "V5"].index(best_overall)) - 0.4,
            y0=0,
            x1=float(["V1", "V2", "V3", "V4", "V5"].index(best_overall)) + 0.4,
            y1=1,
            fillcolor=light_purple,
            opacity=0.15,
            layer="below",
            line_width=0,
        )
        
        # Add "Best Overall" annotation for models
        if len(set([best_dpsperc_model, best_fpr_model, best_teca_model])) == 1:
            # If one model is best at everything
            best_model = best_dpsperc_model
            fig.add_annotation(
                x=best_model,
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
                x=best_overall,
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
                tickvals=["V1", "V2", "V3", "V4", "V5"],
                ticktext=["V1", "V2", "V3", "V4", "V5"],
                tickangle=0,
                tickfont=dict(size=12, color=dark_purple),
                showgrid=False
            ),
            yaxis=dict(
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
            <strong>Best DPSPerc:</strong> {best_dpsperc_model} ({trend_df.loc[trend_df['Model'] == best_dpsperc_model, 'DPSPerc (%)'].values[0]:.2f}%)<br>
            <strong>Best FPR:</strong> {best_fpr_model} ({trend_df.loc[trend_df['Model'] == best_fpr_model, 'FPR (%)'].values[0]:.2f}%)<br>
            <strong>Best TECA:</strong> {best_teca_model} ({trend_df.loc[trend_df['Model'] == best_teca_model, 'TECA (%)'].values[0]:.2f}%)
            </small>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a brief explanation
        if selected_model == "V5":
            st.caption(f"The timeline shows how metrics for {trend_device} have evolved through model versions. Stars indicate best performance for each metric.")
        else:
            st.caption(f"The timeline shows how metrics for {trend_device} have evolved through versions. Try selecting different models to compare their performance.")

    # Sample Composition Table
    st.markdown("### Sample Composition")
    st.markdown("Distribution of positive and negative samples in the test dataset:")

    # Calculate percentages
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
    The dataset has a relatively balanced distribution for AC and EV detection, while PV detection has more positive examples.
    """)

    # Model comparison
    st.markdown("### Model Comparison")

    # Create tabs for different metrics
    metric_tabs = st.tabs(["DPSPerc", "FPR", "TECA"])

    with metric_tabs[0]:  # DPSPerc tab
        # Create bar chart for DPSPerc across models and devices
        dpsperc_data = []
        for model in ["V1", "V2", "V3", "V4", "V5"]:
            for device in device_types:
                dpsperc_data.append({
                    "Model": model,
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
            font=dict(color=dark_purple)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with metric_tabs[1]:  # FPR tab
        # Create bar chart for FPR across models and devices
        fpr_data = []
        for model in ["V1", "V2", "V3", "V4", "V5"]:
            for device in device_types:
                fpr_data.append({
                    "Model": model,
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
            font=dict(color=dark_purple)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with metric_tabs[2]:  # TECA tab
        # Create bar chart for TECA across models and devices
        teca_data = []
        for model in ["V1", "V2", "V3", "V4", "V5"]:
            for device in device_types:
                teca_data.append({
                    "Model": model,
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
            font=dict(color=dark_purple)
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
        st.caption(f"Performance metrics for {selected_model} model across all dimensions")

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
        st.caption(f"Confusion matrix for {selected_device} detection (%) with {selected_model} model")

    # Key findings section
    st.markdown("### Key Findings")
    st.markdown(f"""
    - The **{selected_model}** model shows strong performance across all device types, with PV Usage detection being particularly accurate.
    - EV Charging detection shows a good balance between DPSPerc ({models_data[selected_model]['DPSPerc']['EV Charging']:.2f}%) and false positives ({models_data[selected_model]['FPR']['EV Charging'] * 100:.2f}%).
    - PV Usage detection achieves the highest DPSPerc ({models_data[selected_model]['DPSPerc']['PV Usage']:.2f}%) among all device types.
    """)

    # Add model comparison insights
    if selected_model == "V4":
        st.markdown("""
        **Model Comparison Insights:**
        - The V4 model achieves the highest EV Charging DPSPerc (85.81%) with the lowest FPR (11.76%), offering the most accurate EV detection.
        - AC Usage detection shows comparable performance across models, with the V4 model providing the best balance of accuracy and false positives.
        - All models demonstrate similar PV detection capabilities, with minor variations in performance metrics.
        """)
    elif selected_model == "V5":
        st.markdown("""
        **Model Comparison Insights:**
        - The V5 model offers improved AC Usage detection with a DPSPerc of 77.58%, higher than previous models.
        - For EV Charging, V5 provides a balanced approach with a DPSPerc of 81.11% and moderate FPR of 13.73%, making it more reliable than earlier versions but not as aggressive as V4.
        - PV Usage detection in V5 (91.93% DPSPerc) remains strong and consistent with previous models.
        - The TECA scores show that V5 achieves good energy assignment accuracy for AC Usage (0.6912) and PV Usage (0.7680), though slightly lower for EV Charging (0.6697) compared to V4.
        - Overall, V5 represents a more balanced model that prioritizes consistent performance across all device types rather than optimizing for any single metric.
        """)

    # Footer (using the same styling as the main page)
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; color:{primary_purple}; padding: 10px; border-radius: 5px;">
        This dashboard presents the performance metrics of our NILM algorithm for detecting various device usage patterns.
        These results help guide model selection and identify areas for further improvement.
    </div>
    """, unsafe_allow_html=True)
