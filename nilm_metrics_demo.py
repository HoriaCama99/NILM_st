import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from PIL import Image
import seaborn as sns

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
        
        # Set region-specific patterns
        region_characteristics = {
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
        
        # Generate data for each state
        for region, states in regions.items():
            char = region_characteristics[region]
            
            for state in states:
                # Add some realistic variance within regions
                variance_factor = random.uniform(0.8, 1.2)
                
                # Base metrics
                grid_consumption = random.uniform(*char['grid_range']) * variance_factor
                solar_production = random.uniform(*char['solar_range']) * variance_factor
                ev_charging = random.uniform(*char['ev_range']) * variance_factor
                ac_usage = random.uniform(*char['ac_range']) * variance_factor
                water_heater = random.uniform(*char['water_heater_range']) * variance_factor
                
                # Adoption rates (percentages)
                pv_adoption = random.uniform(*char['pv_adoption_range'])
                ev_adoption = random.uniform(*char['ev_adoption_range'])
                
                # Home counts for context
                home_count = int(random.uniform(500, 5000))
                
                # Calculate derived metrics
                solar_coverage = (solar_production / grid_consumption * 100) if grid_consumption > 0 else 0
                ev_percentage = (ev_charging / grid_consumption * 100) if grid_consumption > 0 else 0
                ac_percentage = (ac_usage / grid_consumption * 100) if grid_consumption > 0 else 0
                
                # Energy efficiency score (lower is better) - scaled between 0-100
                efficiency_score = 100 - (grid_consumption - 600) / 8  # Normalize to 0-100 scale
                efficiency_score = max(0, min(100, efficiency_score))  # Clamp to 0-100
                
                geo_data.append({
                    'state': state,
                    'region': region,
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
    
    # Create a two-column layout for map controls and explanations
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
        
        # Filter data by selected regions
        if selected_regions:
            filtered_geo_df = geo_df[geo_df['region'].isin(selected_regions)]
        else:
            filtered_geo_df = geo_df
        
        # Add explanatory text about the selected metric
        metric_explanations = {
            "Grid Consumption (kWh)": "Average monthly electricity consumption from the grid per household.",
            "Solar Production (kWh)": "Average monthly solar energy production per household with PV systems.",
            "EV Charging (kWh)": "Average monthly electricity used for EV charging in homes with EVs.",
            "AC Usage (kWh)": "Average monthly electricity consumed by air conditioning systems.",
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
        
        # Add summary statistics for the selected metric
        metric_col = map_metric.split(" (")[0].lower().replace(" ", "_")
        if metric_col in filtered_geo_df.columns:
            avg_value = filtered_geo_df[metric_col].mean()
            min_value = filtered_geo_df[metric_col].min()
            max_value = filtered_geo_df[metric_col].max()
            
            st.markdown("### Summary Statistics")
            st.markdown(f"**Average:** {avg_value:.1f}")
            st.markdown(f"**Min:** {min_value:.1f}")
            st.markdown(f"**Max:** {max_value:.1f}")
            st.markdown(f"**Range:** {max_value - min_value:.1f}")
    
    with map_col2:
        # Set up the color scales for different metrics
        color_scales = {
            "Grid Consumption (kWh)": [light_purple, primary_purple, dark_purple],
            "Solar Production (kWh)": ["#F9F0D9", cream, green],
            "EV Charging (kWh)": ["#D9DCFF", light_purple, primary_purple],
            "AC Usage (kWh)": ["#D9F2EC", green, "#43867F"],
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
            "PV Adoption Rate (%)": "pv_adoption_rate",
            "EV Adoption Rate (%)": "ev_adoption_rate",
            "Energy Efficiency Score": "efficiency_score"
        }
        
        selected_metric = metric_mappings[map_metric]
        
        # Create the map visualization
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
                'home_count': 'Homes'
            }
        )
        
        # Update map layout
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
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Add regional comparison section
    if selected_regions:
        st.subheader("Regional Comparison")
        
        # Group data by region and calculate averages
        region_comparison = filtered_geo_df.groupby('region')[
            ['grid_consumption', 'solar_production', 'ev_charging', 
             'ac_usage', 'pv_adoption_rate', 'ev_adoption_rate', 'efficiency_score']
        ].mean().reset_index()
        
        # Create a bar chart comparing regions
        fig = px.bar(
            region_comparison,
            x='region',
            y=selected_metric,
            color='region',
            color_discrete_map={
                'Northeast': primary_purple,
                'Southeast': green,
                'Midwest': light_purple,
                'Southwest': cream,
                'West': salmon
            },
            text=selected_metric,
            labels={selected_metric: map_metric, 'region': 'Region'}
        )
        
        fig.update_layout(
            xaxis_title="Region",
            yaxis_title=map_metric,
            showlegend=False,
            paper_bgcolor=white,
            plot_bgcolor=white,
            font=dict(color=dark_purple)
        )
        
        fig.update_traces(
            texttemplate='%{y:.1f}',
            textposition='outside'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Regional Insights"):
            # Generate insights based on the data
            highest_region = region_comparison.loc[region_comparison[selected_metric].idxmax()]['region']
            lowest_region = region_comparison.loc[region_comparison[selected_metric].idxmin()]['region']
            
            st.markdown(f"""
            ### Key Regional Patterns
            
            - **{highest_region}** shows the highest {map_metric.lower()} ({region_comparison[selected_metric].max():.1f})
            - **{lowest_region}** shows the lowest {map_metric.lower()} ({region_comparison[selected_metric].min():.1f})
            - The difference between highest and lowest regions is {region_comparison[selected_metric].max() - region_comparison[selected_metric].min():.1f}
            
            ### Potential Factors
            
            Different regions show varying patterns due to:
            - Climate differences affecting heating and cooling needs
            - State-level policies and incentives for renewable energy
            - Regional economic factors influencing technology adoption
            - Infrastructure development supporting EV adoption
            - Cultural and demographic differences affecting energy consumption
            """)

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
                    'PV Usage': 92.5777,
                    'WH Usage': 29.2893
                },
                'FPR': {
                    'EV Charging': 0.1961,
                    'AC Usage': 0.3488,
                    'PV Usage': 0.1579,
                    'WH Usage': 0.0000
                },
                'TECA': {
                    'EV Charging': 0.7185,
                    'AC Usage': 0.6713,
                    'PV Usage': 0.8602,
                    'WH Usage': 0.3000
                }
            },
            'V2': {
                'DPSPerc': {
                    'EV Charging': 82.4146,
                    'AC Usage': 76.6360,
                    'PV Usage': 92.5777,
                    'WH Usage': 29.2893
                },
                'FPR': {
                    'EV Charging': 0.1667,
                    'AC Usage': 0.3488,
                    'PV Usage': 0.1579,
                    'WH Usage': 0.0000
                },
                'TECA': {
                    'EV Charging': 0.7544,
                    'AC Usage': 0.6519,
                    'PV Usage': 0.8812,
                    'WH Usage': 0.3107
                }
            },
            'V3': {
                'DPSPerc': {
                    'EV Charging': 81.8612,
                    'AC Usage': 73.0220,
                    'PV Usage': 92.0629,
                    'WH Usage': 29.2893
                },
                'FPR': {
                    'EV Charging': 0.1667,
                    'AC Usage': 0.4419,
                    'PV Usage': 0.1754,
                    'WH Usage': 0.0000
                },
                'TECA': {
                    'EV Charging': 0.7179,
                    'AC Usage': 0.6606,
                    'PV Usage': 0.8513,
                    'WH Usage': 0.0582
                }
            },
            'V4': {
                'DPSPerc': {
                    'EV Charging': 85.8123,
                    'AC Usage': 76.3889,
                    'PV Usage': 91.5448,
                    'WH Usage': 29.2893
                },
                'FPR': {
                    'EV Charging': 0.1176,
                    'AC Usage': 0.1977,
                    'PV Usage': 0.1930,
                    'WH Usage': 0.0000
                },
                'TECA': {
                    'EV Charging': 0.7741,
                    'AC Usage': 0.6718,
                    'PV Usage': 0.8395,
                    'WH Usage': -0.0181
                }
            }
        }
        
        # Confusion matrix is updated later based on selected model and device
        
        return models_data

    # Load performance data
    models_data = load_performance_data()

    # Model selection in sidebar
    selected_model = st.sidebar.selectbox(
        "Select Model",
        ["V1", "V2", "V3", "V4"],
        index=3,  # Default to V4
        format_func=lambda x: x  # No need to transform the names anymore
    )

    # Add device type filter in sidebar
    device_types = st.sidebar.multiselect(
        "Select Device Types",
        ["EV Charging", "AC Usage", "PV Usage", "WH Usage"],
        default=["EV Charging", "AC Usage", "PV Usage", "WH Usage"]
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

    # Key metrics display section
    st.markdown("### Key Metrics")

    # Create metrics in 4-column layout
    metrics_cols = st.columns(4)

    # Define colors for each device
    device_colors = {
        'EV Charging': primary_purple,
        'AC Usage': green,
        'PV Usage': cream,
        'WH Usage': salmon
    }

    # Display metric cards for each device type
    for i, device in enumerate(device_types):
        with metrics_cols[i % 4]:
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
                f"{fpr_value:.4f}",
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

    # Sample Composition Table
    st.markdown("### Sample Composition")
    st.markdown("Distribution of positive and negative samples in the test dataset:")

    # Calculate percentages
    sample_composition = {
        'Device Type': ['EV Charging', 'AC Usage', 'PV Usage', 'WH Usage'],
        'Negative Class (%)': [64.56, 54.43, 36.08, 91.14],
        'Positive Class (%)': [35.44, 45.57, 63.92, 8.86]
    }

    # Create a DataFrame
    sample_df = pd.DataFrame(sample_composition)

    # Add count information in tooltips
    sample_df['Negative Count'] = [102, 86, 57, 144]
    sample_df['Positive Count'] = [56, 72, 101, 14]

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
    WH Usage detection has a significant class imbalance ({sample_df['Positive Class (%)'][3]:.2f}% positive),
    which may partially explain the lower performance metrics for this device type.
    """)

    # Model comparison
    st.markdown("### Model Comparison")

    # Create tabs for different metrics
    metric_tabs = st.tabs(["DPSPerc", "FPR", "TECA"])

    with metric_tabs[0]:  # DPSPerc tab
        # Create bar chart for DPSPerc across models and devices
        dpsperc_data = []
        for model in ["V1", "V2", "V3", "V4"]:
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
                "PV Usage": cream,
                "WH Usage": salmon
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
        for model in ["V1", "V2", "V3", "V4"]:
            for device in device_types:
                fpr_data.append({
                    "Model": model,
                    "Device": device,
                    "FPR": models_data[model]["FPR"][device]
                })
        
        fpr_df = pd.DataFrame(fpr_data)
        
        fig = px.bar(
            fpr_df, 
            x="Model", 
            y="FPR", 
            color="Device",
            barmode="group",
            color_discrete_map={
                "EV Charging": primary_purple,
                "AC Usage": green,
                "PV Usage": cream,
                "WH Usage": salmon
            },
            template="plotly_white"
        )
        
        fig.update_layout(
            title="False Positive Rate Comparison Across Models (Lower is Better)",
            xaxis_title="Model",
            yaxis_title="FPR",
            legend_title="Device",
            paper_bgcolor=white,
            plot_bgcolor=white,
            font=dict(color=dark_purple)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with metric_tabs[2]:  # TECA tab
        # Create bar chart for TECA across models and devices
        teca_data = []
        for model in ["V1", "V2", "V3", "V4"]:
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
                "PV Usage": cream,
                "WH Usage": salmon
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
        elif selected_device == "PV Usage":
            labels = ['No PV', 'PV Detected']
            title = 'PV Usage Detection'
            cmap = 'Oranges'
        else:  # WH Usage
            labels = ['No WH', 'WH Detected']
            title = 'WH Usage Detection'
            cmap = 'Reds'

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
    - The **{selected_model}** model shows strong performance across most device types, with PV Usage detection being particularly accurate.
    - EV Charging detection shows excellent balance between DPSPerc ({models_data[selected_model]['DPSPerc']['EV Charging']:.2f}%) and low false positives ({models_data[selected_model]['FPR']['EV Charging']:.4f}).
    - WH Usage detection remains challenging with lower DPSPerc, suggesting further model improvements are needed for this specific device type.
    - PV Usage detection achieves the highest DPSPerc ({models_data[selected_model]['DPSPerc']['PV Usage']:.2f}%) and TECA ({models_data[selected_model]['TECA']['PV Usage'] * 100:.2f}%) among all device types.
    """)

    # Add model comparison insights
    if selected_model == "V4":
        st.markdown("""
        **Model Comparison Insights:**
        - The V4 model achieves the highest EV Charging DPSPerc (85.81%) with the lowest FPR (0.1176), offering the most accurate EV detection.
        - AC Usage detection shows comparable performance across models, with the V4 model providing the best balance of accuracy and false positives.
        - All models demonstrate similar PV detection capabilities, with minor variations in performance metrics.
        """)

    # Footer (using the same styling as the main page)
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; color:{primary_purple}; padding: 10px; border-radius: 5px;">
        This dashboard presents the performance metrics of our NILM algorithm for detecting various device usage patterns.
        These results help guide model selection and identify areas for further improvement.
    </div>
    """, unsafe_allow_html=True)
