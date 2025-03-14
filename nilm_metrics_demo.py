import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from PIL import Image
import seaborn as sns
import random

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
        total_identified = (filtered_df['ev charging (kWh)'].sum() + 
                          filtered_df['air conditioning (kWh)'].sum() + 
                          filtered_df['water heater (kWh)'].sum())
        
        pct_identified = (total_identified / total_grid * 100) if total_grid > 0 else 0
        
        st.metric(
            label="Consumption Identified",
            value=f"{pct_identified:.1f}%",
            help="Percentage of total grid consumption attributed to specific appliances"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Add Geographic Map Section - NEW SECTION
    st.subheader("Geographic Performance Distribution")

    # Create tabs for different map views
    map_tabs = st.tabs(["Device Adoption", "Model Performance", "Energy Consumption"])

    # Generate random coordinates across the US
    num_points = 100

    # Define regions with their approximate bounds
    regions = {
        "Northeast": {"lat_range": (40, 47), "lon_range": (-80, -67)},
        "Midwest": {"lat_range": (36, 49), "lon_range": (-97, -80)},
        "South": {"lat_range": (25, 36), "lon_range": (-106, -75)},
        "West": {"lat_range": (32, 49), "lon_range": (-124, -107)}
    }

    # Generate mock data with regional biases
    mock_geo_data = []

    for region_name, bounds in regions.items():
        # Number of points per region
        points_in_region = num_points // 4
        
        for _ in range(points_in_region):
            lat = random.uniform(bounds["lat_range"][0], bounds["lat_range"][1])
            lon = random.uniform(bounds["lon_range"][0], bounds["lon_range"][1])
            
            # Create regional biases in the data
            if region_name == "West":
                ev_adoption = random.uniform(0.15, 0.35)  # Higher EV adoption in West
                pv_adoption = random.uniform(0.20, 0.40)  # Higher solar in West
                ac_adoption = random.uniform(0.60, 0.85)
                wh_adoption = random.uniform(0.30, 0.50)
                model_accuracy = random.uniform(0.75, 0.90)
            elif region_name == "Northeast":
                ev_adoption = random.uniform(0.10, 0.25)
                pv_adoption = random.uniform(0.05, 0.20)  # Lower solar in Northeast
                ac_adoption = random.uniform(0.50, 0.75)
                wh_adoption = random.uniform(0.40, 0.60)  # Higher water heater in Northeast
                model_accuracy = random.uniform(0.70, 0.85)
            elif region_name == "Midwest":
                ev_adoption = random.uniform(0.05, 0.15)  # Lower EV in Midwest
                pv_adoption = random.uniform(0.05, 0.15)
                ac_adoption = random.uniform(0.70, 0.90)  # Higher AC in Midwest
                wh_adoption = random.uniform(0.35, 0.55)
                model_accuracy = random.uniform(0.65, 0.80)
            else:  # South
                ev_adoption = random.uniform(0.08, 0.20)
                pv_adoption = random.uniform(0.10, 0.25)
                ac_adoption = random.uniform(0.80, 0.95)  # Highest AC in South
                wh_adoption = random.uniform(0.25, 0.45)
                model_accuracy = random.uniform(0.70, 0.85)
            
            # Add some random variation to make it look more realistic
            ev_detected = 1 if random.random() < ev_adoption else 0
            ac_detected = 1 if random.random() < ac_adoption else 0
            pv_detected = 1 if random.random() < pv_adoption else 0
            wh_detected = 1 if random.random() < wh_adoption else 0
            
            # Calculate mock energy values
            ev_energy = random.uniform(50, 200) if ev_detected else 0
            ac_energy = random.uniform(100, 400) if ac_detected else 0
            pv_energy = random.uniform(200, 600) if pv_detected else 0
            wh_energy = random.uniform(50, 150) if wh_detected else 0
            
            # Add to dataset
            mock_geo_data.append({
                "latitude": lat,
                "longitude": lon,
                "region": region_name,
                "ev_adoption": ev_adoption,
                "ac_adoption": ac_adoption,
                "pv_adoption": pv_adoption,
                "wh_adoption": wh_adoption,
                "ev_detected": ev_detected,
                "ac_detected": ac_detected,
                "pv_detected": pv_detected,
                "wh_detected": wh_detected,
                "ev_energy": ev_energy,
                "ac_energy": ac_energy,
                "pv_energy": pv_energy,
                "wh_energy": wh_energy,
                "model_accuracy": model_accuracy
            })

    # Convert to DataFrame
    mock_geo_df = pd.DataFrame(mock_geo_data)

    with map_tabs[0]:  # Device Adoption tab
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Add region selector
            selected_region = st.selectbox(
                "Select Region", 
                ["All Regions", "Northeast", "Midwest", "South", "West"],
                key="map_region_selector"
            )
            
            # Add device type selector for the map
            map_device = st.selectbox(
                "Select Device Type", 
                ["EV Charging", "AC Usage", "PV Usage", "WH Usage", "All Devices"],
                key="map_device_selector"
            )
            
            # Add a brief explanation
            st.markdown("""
            This map shows the geographic distribution of device adoption rates across different regions.
            
            - Larger circles indicate higher adoption rates
            - Color intensity corresponds to adoption percentage
            - Hover over points to see detailed information
            """)
        
        with col2:
            # Filter data based on region selection
            if selected_region != "All Regions":
                display_geo_df = mock_geo_df[mock_geo_df["region"] == selected_region]
            else:
                display_geo_df = mock_geo_df
            
            # Determine which data to show based on device selection
            if map_device == "EV Charging":
                color_data = 'ev_adoption'
                size_data = 'ev_adoption'
                color_scale = [[0, primary_purple], [1, light_purple]]
                title = "EV Charging Adoption Rate"
            elif map_device == "AC Usage":
                color_data = 'ac_adoption'
                size_data = 'ac_adoption'
                color_scale = [[0, "#B5E2C9"], [1, green]]
                title = "AC Usage Adoption Rate"
            elif map_device == "PV Usage":
                color_data = 'pv_adoption'
                size_data = 'pv_adoption'
                color_scale = [[0, "#F9EFD6"], [1, cream]]
                title = "PV Usage Adoption Rate"
            elif map_device == "WH Usage":
                color_data = 'wh_adoption'
                size_data = 'wh_adoption'
                color_scale = [[0, "#F5D0C5"], [1, salmon]]
                title = "WH Usage Adoption Rate"
            else:
                # For "All Devices", use model accuracy as color and total devices as size
                color_data = 'model_accuracy'
                # Calculate total devices detected for size
                display_geo_df['total_devices'] = display_geo_df['ev_detected'] + display_geo_df['ac_detected'] + display_geo_df['pv_detected'] + display_geo_df['wh_detected']
                size_data = 'total_devices'
                color_scale = [[0, "#D0D0D0"], [1, primary_purple]]
                title = "Overall Device Adoption"
            
            # Create the map
            fig = px.scatter_mapbox(
                display_geo_df,
                lat="latitude",
                lon="longitude",
                color=color_data,
                size=size_data,
                size_max=15,
                hover_name="region",
                hover_data={
                    "latitude": False,
                    "longitude": False,
                    "ev_adoption": ':.1%',
                    "ac_adoption": ':.1%',
                    "pv_adoption": ':.1%',
                    "wh_adoption": ':.1%',
                    "model_accuracy": ':.1%'
                },
                color_continuous_scale=color_scale,
                zoom=3.5 if selected_region == "All Regions" else 5,
                mapbox_style="carto-positron",
                title=title
            )
            
            fig.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                coloraxis_colorbar=dict(
                    title="Adoption Rate",
                    tickformat='.0%'
                ),
                height=500,
                paper_bgcolor=white,
                plot_bgcolor=white,
                font=dict(color=dark_purple)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with map_tabs[1]:  # Model Performance tab
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Add region selector
            perf_region = st.selectbox(
                "Select Region", 
                ["All Regions", "Northeast", "Midwest", "South", "West"],
                key="perf_region_selector"
            )
            
            # Add metric selector
            perf_metric = st.selectbox(
                "Select Performance Metric", 
                ["Model Accuracy", "DPSPerc", "FPR", "TECA"],
                key="perf_metric_selector"
            )
            
            # Add a brief explanation
            st.markdown("""
            This map visualizes model performance metrics across different regions.
            
            - Color intensity shows performance level
            - Larger circles indicate higher confidence
            - Regional patterns may suggest areas for model improvement
            """)
        
        with col2:
            # Filter data based on region selection
            if perf_region != "All Regions":
                perf_geo_df = mock_geo_df[mock_geo_df["region"] == perf_region]
            else:
                perf_geo_df = mock_geo_df
            
            # Add mock performance metrics if they don't exist
            if "dpsperc" not in perf_geo_df.columns:
                perf_geo_df["dpsperc"] = perf_geo_df["model_accuracy"] * random.uniform(0.9, 1.1)
                perf_geo_df["fpr"] = 1 - (perf_geo_df["model_accuracy"] * random.uniform(0.8, 1.0))
                perf_geo_df["teca"] = perf_geo_df["model_accuracy"] * random.uniform(0.85, 1.15)
                # Ensure values are in reasonable ranges
                perf_geo_df["dpsperc"] = perf_geo_df["dpsperc"].clip(0.5, 0.95)
                perf_geo_df["fpr"] = perf_geo_df["fpr"].clip(0.05, 0.5)
                perf_geo_df["teca"] = perf_geo_df["teca"].clip(0.4, 0.9)
            
            # Determine which metric to show
            if perf_metric == "Model Accuracy":
                color_data = 'model_accuracy'
                title = "Overall Model Accuracy"
                color_range = [0.6, 0.9]
            elif perf_metric == "DPSPerc":
                color_data = 'dpsperc'
                title = "Distance to Perfect Score (DPSPerc)"
                color_range = [0.5, 0.95]
            elif perf_metric == "FPR":
                color_data = 'fpr'
                title = "False Positive Rate (FPR)"
                color_range = [0.05, 0.5]
                # Invert color scale for FPR (lower is better)
                color_scale = [[0, green], [1, red]]
            else:  # TECA
                color_data = 'teca'
                title = "Total Energy Correctly Assigned (TECA)"
                color_range = [0.4, 0.9]
            
            # Set default color scale (except for FPR which is defined above)
            if perf_metric != "FPR":
                color_scale = [[0, "#D0D0D0"], [1, primary_purple]]
            
            # Create the map
            fig = px.scatter_mapbox(
                perf_geo_df,
                lat="latitude",
                lon="longitude",
                color=color_data,
                size='model_accuracy',  # Use accuracy for size to show confidence
                size_max=15,
                hover_name="region",
                hover_data={
                    "latitude": False,
                    "longitude": False,
                    "model_accuracy": ':.1%',
                    "dpsperc": ':.1%',
                    "fpr": ':.1%',
                    "teca": ':.1%'
                },
                color_continuous_scale=color_scale,
                range_color=color_range,
                zoom=3.5 if perf_region == "All Regions" else 5,
                mapbox_style="carto-positron",
                title=title
            )
            
            fig.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                coloraxis_colorbar=dict(
                    title=perf_metric,
                    tickformat='.0%'
                ),
                height=500,
                paper_bgcolor=white,
                plot_bgcolor=white,
                font=dict(color=dark_purple)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with map_tabs[2]:  # Energy Consumption tab
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Add region selector
            energy_region = st.selectbox(
                "Select Region", 
                ["All Regions", "Northeast", "Midwest", "South", "West"],
                key="energy_region_selector"
            )
            
            # Add energy type selector
            energy_type = st.selectbox(
                "Select Energy Type", 
                ["Total Consumption", "EV Charging", "AC Usage", "PV Production", "WH Usage"],
                key="energy_type_selector"
            )
            
            # Add a brief explanation
            st.markdown("""
            This map shows energy consumption patterns across different regions.
            
            - Color intensity indicates energy usage levels
            - Larger circles represent higher consumption
            - Regional patterns may reflect climate differences and adoption rates
            """)
        
        with col2:
            # Filter data based on region selection
            if energy_region != "All Regions":
                energy_geo_df = mock_geo_df[mock_geo_df["region"] == energy_region]
            else:
                energy_geo_df = mock_geo_df
            
            # Calculate total energy for each point
            energy_geo_df["total_energy"] = energy_geo_df["ev_energy"] + energy_geo_df["ac_energy"] + energy_geo_df["wh_energy"]
            
            # Determine which energy data to show
            if energy_type == "Total Consumption":
                color_data = 'total_energy'
                size_data = 'total_energy'
                title = "Total Energy Consumption (kWh)"
                color_scale = [[0, "#D0D0D0"], [1, primary_purple]]
            elif energy_type == "EV Charging":
                color_data = 'ev_energy'
                size_data = 'ev_energy'
                title = "EV Charging Energy (kWh)"
                color_scale = [[0, "#D0D0D0"], [1, primary_purple]]
            elif energy_type == "AC Usage":
                color_data = 'ac_energy'
                size_data = 'ac_energy'
                title = "AC Usage Energy (kWh)"
                color_scale = [[0, "#D0D0D0"], [1, green]]
            elif energy_type == "PV Production":
                color_data = 'pv_energy'
                size_data = 'pv_energy'
                title = "Solar PV Production (kWh)"
                color_scale = [[0, "#D0D0D0"], [1, cream]]
            else:  # WH Usage
                color_data = 'wh_energy'
                size_data = 'wh_energy'
                title = "Water Heater Energy (kWh)"
                color_scale = [[0, "#D0D0D0"], [1, salmon]]
            
            # Create the map
            fig = px.scatter_mapbox(
                energy_geo_df,
                lat="latitude",
                lon="longitude",
                color=color_data,
                size=size_data,
                size_max=15,
                hover_name="region",
                hover_data={
                    "latitude": False,
                    "longitude": False,
                    "total_energy": ':.1f',
                    "ev_energy": ':.1f',
                    "ac_energy": ':.1f',
                    "pv_energy": ':.1f',
                    "wh_energy": ':.1f'
                },
                color_continuous_scale=color_scale,
                zoom=3.5 if energy_region == "All Regions" else 5,
                mapbox_style="carto-positron",
                title=title
            )
            
            fig.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                coloraxis_colorbar=dict(
                    title="Energy (kWh)"
                ),
                height=500,
                paper_bgcolor=white,
                plot_bgcolor=white,
                font=dict(color=dark_purple)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Add a note about the mock data
    st.caption("Note: This map uses mock data for demonstration purposes. In a production environment, it would display actual geographic data from your dataset.")

    # Add visualizations
    st.subheader("Energy Consumption Visualizations")

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
