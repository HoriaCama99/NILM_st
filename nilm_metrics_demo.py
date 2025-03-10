import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Energy Disaggregation Model Output",
    page_icon="⚡",
    layout="wide"
)

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

# Banner with hardcoded path
banner_path = "ECMX_linkedinheader_SN.png" 

try:
    # Display banner image
    banner_image = Image.open(banner_path)
    st.image(banner_image, use_container_width=True)
except Exception as e:
    st.warning(f"Banner image not found at {banner_path}. Please update the path in the code.")

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
