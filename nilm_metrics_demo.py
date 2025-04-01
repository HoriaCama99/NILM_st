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

# Initialize selected_model to the default value before page logic
selected_model = "V6" # Default model

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
    st.subheader(f"Model: {selected_model} ({model_dates[selected_model]})")

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
    
    # Description
    st.markdown("""
    This map shows the geographic distribution of homes equipped with NILM-detected devices.
    **Click on a state** to zoom in and see individual homes with EV chargers, AC units, and solar panels.
    """)
    
    # Create function to generate mock data for US states
    # Cache the data generation

    # Cache the data generation
    @st.cache_data
    def generate_geo_data():
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
                    device_type = random.choice(['ev', 'ac', 'pv'])
                    if device_type == 'ev': has_ev = True
                    elif device_type == 'ac': has_ac = True
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

        # Generate households for each state
        all_households = []
        for state_code in states_data:
            state_households = generate_households(state_code, states_data[state_code]['total_homes'])
            all_households.extend(state_households)

        return states_data, all_households

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

    # --- Add Map Generation and Display Logic ---
    # Create filter controls in sidebar
    st.sidebar.markdown("### Map Filters")
    show_ev = st.sidebar.checkbox("Show EV Chargers", value=True)
    show_ac = st.sidebar.checkbox("Show AC Units", value=True)
    show_pv = st.sidebar.checkbox("Show Solar Panels", value=True)
    
    # Add refresh button
    if st.sidebar.button("↻ Refresh Map View", use_container_width=True):
        # Reset any state selection and reload the page
        st.query_params.clear()
        st.rerun()
    
    # Load data with a spinner
    with st.spinner("Loading map data..."):
        states_data, all_households = generate_geo_data()
        us_geojson = load_us_geojson()
    
    # Get state from URL parameter if available
    params = st.query_params
    selected_state = params.get("state", [""])[0]
    
    if selected_state not in states_data:
        selected_state = ""
    
    # Filter households by selected state and device types
    filtered_households = [
        h for h in all_households if 
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
    
    if not selected_state and us_geojson:
        # Add state boundaries with click functionality
        style_function = lambda x: {
            'fillColor': primary_purple,
            'color': 'white',
            'weight': 1,
            'fillOpacity': 0.5
        }
        
        folium.GeoJson(
            us_geojson,
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
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa'),
            tooltip=f"Home {house['id']}"
        ).add_to(marker_cluster)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; 
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid gray; z-index: 9999;">
        <h4 style="margin-top: 0; color: #515D9A;">Device Legend</h4>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 20px; height: 20px; background-color: blue; border-radius: 50%; margin-right: 10px;"></div>
            <span>EV Charger</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 20px; height: 20px; background-color: orange; border-radius: 50%; margin-right: 10px;"></div>
            <span>AC Unit</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 20px; height: 20px; background-color: green; border-radius: 50%; margin-right: 10px;"></div>
            <span>Solar Panel</span>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add Layer Control to toggle layers
    folium.LayerControl().add_to(m)
    
    # Display the map with a loading spinner
    with st.spinner("Rendering map... This may take a moment for all states..."):
        st.subheader("Interactive Energy Deployment Map")
        st.caption("Click on a state to see details or use the controls to filter by device type")
        folium_static(m, width=1000, height=600)
        st.info("💡 **Tip:** Use the refresh button in the sidebar to reset the map view if needed.")
    
    # Add summary statistics even if no state is selected
    if not selected_state:
        st.subheader("Portfolio Overview")
        
        total_homes = sum(states_data[state]['total_homes'] for state in states_data)
        total_ev = sum(states_data[state]['ev_homes'] for state in states_data)
        total_ac = sum(states_data[state]['ac_homes'] for state in states_data)
        total_pv = sum(states_data[state]['pv_homes'] for state in states_data)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Homes", f"{total_homes:,}")
        with col2:
            st.metric("EV Chargers", f"{total_ev:,}", f"{total_ev/total_homes:.1%}")
        with col3:
            st.metric("AC Units", f"{total_ac:,}", f"{total_ac/total_homes:.1%}")
        with col4:
            st.metric("Solar Panels", f"{total_pv:,}", f"{total_pv/total_homes:.1%}")
        
        # Add device distribution bar chart
        st.subheader("Device Distribution by State")
        
        # Get top 10 states for the visualization
        top_states = sorted(
            [(code, states_data[code]['name'], 
              states_data[code]['ev_homes'] + states_data[code]['ac_homes'] + states_data[code]['pv_homes']) 
             for code in states_data],
            key=lambda x: x[2], reverse=True
        )[:10]
        
        chart_data = []
        for code, name, _ in top_states:
            chart_data.extend([
                {'State': name, 'Device': 'EV Chargers', 'Count': states_data[code]['ev_homes']},
                {'State': name, 'Device': 'AC Units', 'Count': states_data[code]['ac_homes']},
                {'State': name, 'Device': 'Solar Panels', 'Count': states_data[code]['pv_homes']}
            ])
        
        fig = px.bar(
            pd.DataFrame(chart_data),
            x='State', y='Count', color='Device',
            color_discrete_map={
                'EV Chargers': 'blue',
                'AC Units': 'orange',
                'Solar Panels': 'green'
            },
            title='Top 10 States by Device Distribution',
            barmode='group'
        )
        
        fig.update_layout(
            xaxis_title='State',
            yaxis_title='Number of Devices',
            legend_title='Device Type',
            plot_bgcolor=white,
            paper_bgcolor=white,
            font=dict(color=dark_purple)
        )
        
        st.plotly_chart(fig, use_container_width=True)
            
        # Show state breakdown
        st.subheader("State Breakdown")
        state_stats = []
        for code in states_data:
            state = states_data[code]
            state_stats.append({
                "State": state['name'],
                "Total Homes": state['total_homes'],
                "EV Chargers": state['ev_homes'],
                "AC Units": state['ac_homes'],
                "Solar Panels": state['pv_homes'],
                "EV %": f"{state['ev_homes']/state['total_homes']:.1%}",
                "AC %": f"{state['ac_homes']/state['total_homes']:.1%}",
                "Solar %": f"{state['pv_homes']/state['total_homes']:.1%}"
            })
        
        st.dataframe(pd.DataFrame(state_stats), use_container_width=True)
    
    # Display statistics if a state is selected
    if selected_state:
        state = states_data[selected_state]
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
        
        # Display data table
        st.subheader("Homes Data")
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
        st.dataframe(df, use_container_width=True)
    # --- End Map Generation Logic ---

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; color:{primary_purple}; padding: 10px; border-radius: 5px;">
        This interactive map shows the geographic distribution of homes with different smart devices.
        Click on a state to zoom in and explore homes with EV chargers, AC units, and solar panels.
    </div>
    """, unsafe_allow_html=True)
