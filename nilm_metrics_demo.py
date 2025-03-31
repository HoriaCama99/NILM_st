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
from folium.plugins import MarkerCluster
from PIL import Image
import requests
from branca.element import Element

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
page = st.sidebar.radio("Select Page", ["Sample Output", "Performance Metrics", "Interactive Map"], index=2)

# Display banner on all pages
try:
    # Display banner image
    banner_image = Image.open(banner_path)
    st.image(banner_image, use_container_width=True)
except FileNotFoundError:
    st.warning(f"Banner image not found at {banner_path}. Please check the path.")
except Exception as e:
    st.warning(f"Could not load banner image: {e}")

if page == "Sample Output":
    # Sample Output page code goes here
    st.title("Sample Output")
    st.subheader("Example NILM Device Detection Results")
    
    # Sample output code and visualization
    st.markdown("""
    This page shows sample outputs from our NILM algorithm, demonstrating its ability to detect and disaggregate
    various devices from the total energy consumption signal.
    """)
    
    # Placeholder for sample output visualization
    st.info("Sample visualizations of device detection would be shown here - PLACEHOLDER UNTIL MAP TESTS WILL BE FINISHED.")
    
    # Add more sample output content as needed
    
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
    st.title("NILM Deployment Map")
    st.subheader("Geographic Distribution of Homes with Smart Devices")

    # --- Data Loading and Generation Functions --- 

    @st.cache_data
    def load_us_geojson():
        """Loads US States GeoJSON, filtering to standard states and ensuring 'code' property."""
        geojson_url = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json"
        try:
            response = requests.get(geojson_url)
            response.raise_for_status()
            geojson_data = response.json()

            # --- No filtering by default - assume the source contains states --- 
            # valid_state_codes = [...] # We might need a list if the geojson includes territories we want to exclude
            processed_features = []
            all_state_codes = [] # Keep track of codes found in GeoJSON
            for feature in geojson_data.get('features', []):
                 state_id = feature.get('id') # Use the ID from the GeoJSON
                 if state_id and len(state_id) == 2: # Basic check for state codes
                     if 'properties' not in feature: feature['properties'] = {}
                     feature['properties']['code'] = state_id # Ensure 'code' exists, matching the ID
                     if 'name' not in feature['properties']: feature['properties']['name'] = state_id # Fallback name
                     processed_features.append(feature)
                     all_state_codes.append(state_id)

            if not processed_features:
                raise ValueError("No valid state features found in GeoJSON data.")

            geojson_data['features'] = processed_features
            st.session_state['all_state_codes'] = sorted(list(set(all_state_codes))) # Store available codes
            return geojson_data

        except Exception as e: 
            st.error(f"Failed to load/process state boundaries: {e}")
            st.session_state['all_state_codes'] = [] # Ensure it exists even on failure
            return None

    @st.cache_data
    def generate_mock_data(state_codes):
        """Generates mock coordinates, stats, and households for given state codes."""
        if not state_codes:
            return {}, []
            
        # Placeholder coordinates/zoom - replace with real data for accuracy
        # Using a simple dict lookup for known states, default for others
        coord_defaults = {'lat': 39.8, 'lon': -98.6, 'zoom': 4}
        known_coords = {
            'CA': {'lat': 36.8, 'lon': -119.4, 'zoom': 6}, 'TX': {'lat': 31.9, 'lon': -99.9, 'zoom': 6},
            'NY': {'lat': 42.2, 'lon': -74.9, 'zoom': 7}, 'FL': {'lat': 27.7, 'lon': -81.5, 'zoom': 6},
            'IL': {'lat': 40.6, 'lon': -89.4, 'zoom': 7}, 'PA': {'lat': 41.2, 'lon': -77.2, 'zoom': 7},
            'OH': {'lat': 40.4, 'lon': -82.9, 'zoom': 7}, 'MA': {'lat': 42.4, 'lon': -71.4, 'zoom': 8},
            'WA': {'lat': 47.8, 'lon': -120.7, 'zoom': 7}
            # Add more states here or load from a file for full coverage
        }

        states_data_dict = {}
        all_households = []
        
        for code in state_codes:
            coords = known_coords.get(code, coord_defaults)
            # Use code as name if not otherwise specified (GeoJSON should provide better names)
            states_data_dict[code] = {'name': code, 'lat': coords['lat'], 'lon': coords['lon'], 'zoom': coords['zoom']}
            
            # Generate stats and households
            total = random.randint(50, 250) # Reduced count per state for performance
            ev_c = random.randint(5, int(total * 0.2))
            ac_c = random.randint(int(total * 0.4), int(total * 0.8))
            pv_c = random.randint(3, int(total * 0.15))
            states_data_dict[code]['stats'] = {'total': total, 'ev': ev_c, 'ac': ac_c, 'pv': pv_c}
            
            # Generate households around the state center
            lat_spread, lon_spread = 1.5, 1.5 # Adjust spread as needed
            for i in range(total):
                lat = coords['lat'] + (random.random() - 0.5) * lat_spread
                lon = coords['lon'] + (random.random() - 0.5) * lon_spread
                has_ev = random.random() < (ev_c / total)
                has_ac = random.random() < (ac_c / total)
                has_pv = random.random() < (pv_c / total)
                if not (has_ev or has_ac or has_pv): # Ensure at least one
                    dev = random.choice(['ev', 'ac', 'pv'])
                    if dev == 'ev': has_ev = True
                    elif dev == 'ac': has_ac = True
                    else: has_pv = True
                
                all_households.append({
                    'id': f"{code}-{i+1}", 'lat': lat, 'lon': lon, 
                    'has_ev': has_ev, 'has_ac': has_ac, 'has_pv': has_pv, 
                    'energy_consumption': random.randint(20, 100), 'state': code
                })
                
        # Try to update names from GeoJSON if possible (assuming geojson loaded first)
        if 'us_states_geojson' in st.session_state and st.session_state.us_states_geojson:
            geojson = st.session_state.us_states_geojson
            for feature in geojson.get('features', []):
                code = feature.get('properties', {}).get('code')
                name = feature.get('properties', {}).get('name')
                if code in states_data_dict and name:
                    states_data_dict[code]['name'] = name
                    
        return states_data_dict, all_households

    # --- Load Data --- 
    # Load GeoJSON first to get the list of state codes
    us_states_geojson = load_us_geojson()
    st.session_state.us_states_geojson = us_states_geojson # Store for generate_mock_data

    if not us_states_geojson or 'all_state_codes' not in st.session_state or not st.session_state.all_state_codes:
        st.error("Map cannot be displayed: Failed to load or process state boundary data.")
        st.stop()
        
    # Generate mock data based on codes found in GeoJSON
    states_data, households = generate_mock_data(st.session_state.all_state_codes)
    all_state_codes = st.session_state.all_state_codes # Use the sorted list from loader
    state_names_map = {code: states_data[code]['name'] for code in all_state_codes if code in states_data}

    # --- Sidebar Filters --- 
    st.sidebar.markdown("### Map Filters")
    show_ev = st.sidebar.checkbox("Show EV Chargers", value=True, key="map_cb_ev")
    show_ac = st.sidebar.checkbox("Show AC Units", value=True, key="map_cb_ac")
    show_pv = st.sidebar.checkbox("Show Solar Panels", value=True, key="map_cb_pv")

    # --- State Selection Logic --- 
    params = st.query_params
    # state_code_from_click is the state code just clicked (from URL param)
    state_code_from_click = params.get("state", [None])[0]
    # st.write(f"Debug: Param 'state' from URL = {state_code_from_click}") # Debug
    
    # displayed_state_code is the state currently being shown/selected in dropdown
    # Initialize from clicked state if valid, else None
    if 'displayed_state_code' not in st.session_state:
        st.session_state.displayed_state_code = state_code_from_click if state_code_from_click in all_state_codes else None
        # If initialized from click, clear param ONLY AFTER first load potentially?
        # Maybe clear only if different from session state? Seems complex.
        # Let's avoid clearing here for now.
        # if st.session_state.displayed_state_code: st.query_params.clear() 
        
    # If a click happened (query param set) and it's different from current session state, update session state
    elif state_code_from_click and state_code_from_click in all_state_codes and state_code_from_click != st.session_state.displayed_state_code:
         st.session_state.displayed_state_code = state_code_from_click
         # DO NOT clear query params here - let the state update drive the rerun
         # st.query_params.clear() 
         # Rerun might be implicitly needed if Streamlit doesn't pick up session state change quickly enough for map
         # st.rerun() # Let's see if it works without explicit rerun first

    # Determine if we are showing the overview or a specific state's details
    # Use the value directly from session state
    final_display_code = st.session_state.get('displayed_state_code', None)
    show_overview = final_display_code is None
    # st.write(f"Debug: Final display code = {final_display_code}, Show overview = {show_overview}") # Debug

    # --- UI Elements: Back Button, State Selector, Stats --- 
    # state_selector_value = None # No longer needed here
    if not show_overview:
        # current_display_code = st.session_state.displayed_state_code # Use final_display_code
        state_info = states_data.get(final_display_code)
        
        # --- Back Button and State Selector Row --- 
        control_cols = st.columns([1, 3]) 
        with control_cols[0]:
            if st.button("← Back to US Overview", key="back_button_main", type="primary", use_container_width=True):
                st.session_state.displayed_state_code = None # Reset state
                st.query_params.clear() # Clear URL param when going back
                st.rerun()
        
        with control_cols[1]:
            # State Selector Dropdown (only shown when a state is selected)
            try:
                # Ensure current_display_code is valid before finding index
                if final_display_code in all_state_codes:
                     default_index = all_state_codes.index(final_display_code)
                else:
                     # This case should ideally not happen if state is validated before setting session state
                     st.warning(f"State code '{final_display_code}' not found in available codes. Resetting to overview.")
                     st.session_state.displayed_state_code = None
                     st.query_params.clear()
                     st.rerun() # Rerun to go back to overview
                     
            except ValueError: # Should be caught by the check above, but as safety
                default_index = 0 # Fallback
                st.warning(f"Error finding index for state '{final_display_code}'.")

                
            state_selector_value = st.selectbox(
                "Select State to View Details:",
                options=all_state_codes,
                format_func=lambda code: state_names_map.get(code, code), # Show name, fallback to code
                index=default_index,
                key="state_selector"
            )
            # Update session state if dropdown changes
            if state_selector_value != st.session_state.displayed_state_code:
                st.session_state.displayed_state_code = state_selector_value
                # Rerun needed after dropdown change to update map etc.
                st.rerun() 
                
        # --- Display State Statistics --- 
        # Use the *potentially updated* state code after selectbox interaction for stats
        current_display_code_for_stats = st.session_state.displayed_state_code 
        state_info_for_stats = states_data.get(current_display_code_for_stats) 

        if state_info_for_stats:
            state_stats = state_info_for_stats.get('stats', {'total': 0, 'ev': 0, 'ac': 0, 'pv': 0})
            st.markdown(f"### {state_info_for_stats.get('name', current_display_code_for_stats)} Statistics") 
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            total_homes = state_stats.get('total', 0)
            m_col1.metric("Total Homes", f"{total_homes:,}")
            if total_homes > 0:
                 m_col2.metric("EV Homes", f"{state_stats.get('ev', 0):,}", f"{state_stats.get('ev', 0)/total_homes*100:.1f}%")
                 m_col3.metric("AC Homes", f"{state_stats.get('ac', 0):,}", f"{state_stats.get('ac', 0)/total_homes*100:.1f}%")
                 m_col4.metric("PV Homes", f"{state_stats.get('pv', 0):,}", f"{state_stats.get('pv', 0)/total_homes*100:.1f}%")
            else:
                 m_col2.metric("EV Homes", "0", "0.0%")
                 m_col3.metric("AC Homes", "0", "0.0%")
                 m_col4.metric("PV Homes", "0", "0.0%")
        # No warning needed if state_info is None, handled by overview logic

    else:
        # Prompt on overview map
        st.markdown("""<div style='text-align: center; padding: 20px; background-color: rgba(81, 93, 154, 0.05); border-radius: 5px; margin-bottom: 20px;'><h3 style='margin-top: 0; color: #515D9A;'>Select a State on the Map</h3><p>Click directly on any state boundary to select it and view details.</p></div>""", unsafe_allow_html=True)


    # --- Filter Households for Detail Map --- 
    # Use the final determined state code for filtering
    # final_display_code is already determined above based on session state
    # show_overview is also determined above

    # --- Create Folium Map --- 
    st.markdown("### Interactive Map")
    st.markdown("""<div style='background-color: rgba(81, 93, 154, 0.1); padding: 15px; border-radius: 5px; margin-bottom: 15px;'><h4 style='margin-top: 0;'>How to Use This Map</h4><p><strong>Overview:</strong> Click a state boundary to select it.</p><p><strong>State View:</strong> Shows selected state details and homes. Use the dropdown or click 'Back' to change view.</p></div><p><em><strong>Disclaimer:</strong> Map uses synthetic data.</em></p>""", unsafe_allow_html=True)

    # Determine map center and zoom based on view
    if final_display_code and final_display_code in states_data:
         map_center = [states_data[final_display_code]['lat'], states_data[final_display_code]['lon']]
         map_zoom = states_data[final_display_code]['zoom']
    else:
         map_center = [39.8, -98.6] # US center
         map_zoom = 4 # Overview zoom
         
    m = folium.Map(location=map_center, zoom_start=map_zoom, tiles="CartoDB positron", control_scale=True)

    # --- Define JS Navigation Function Globally --- 
    # This function will be called by the onclick event added by the 'script' parameter
    navigation_js = """
        <script>
            function navigateToState(stateCode) {
                if (!stateCode) {
                    console.error('JS Error: navigateToState called without stateCode.');
                    return;
                }
                console.log('JS: navigateToState called for: ' + stateCode);
                // Set the query parameter. Streamlit will detect this on rerun.
                window.location.search = 'state=' + stateCode;
            }
        </script>
    """
    m.get_root().html.add_child(Element(navigation_js))

    # --- Map Layers --- 
    if show_overview: 
        # --- Overview Map Configuration --- 
        style_function = lambda x: {'fillColor': primary_purple, 'color': 'white', 'weight': 1, 'fillOpacity': 0.6}
        highlight_function = lambda x: {'fillColor': light_purple, 'weight': 3, 'fillOpacity': 0.8}

        # --- Define JS Click Script (to be attached per feature) ---
        click_script = """
            function(feature, layer) {
                layer.options.interactive = true; // Ensure layer reacts to events
                const stateCode = feature.properties.code || feature.id;
                const stateName = feature.properties.name || stateCode;

                if (!stateCode) { return; } // Skip if no code/id

                // Attach event listeners
                layer.on({
                    mouseover: function(e) {
                        const layer = e.target;
                        layer.setStyle({ fillOpacity: 0.8, weight: 2 }); // Use highlight style
                        if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {
                            layer.bringToFront();
                        }
                    },
                    mouseout: function(e) {
                        // Reset to default style - GeoJson layer usually handles this
                        // Or explicitly reset: e.target.setStyle({ fillOpacity: 0.6, weight: 1 });
                        const layer = e.target;
                        layer.setStyle({ fillOpacity: 0.6, weight: 1 }); // Explicit reset matching style_function
                    },
                    click: function(e) {
                        console.log(`JS: Click detected on state ${stateCode}`);
                        // Call the globally defined navigation function
                        if (typeof navigateToState === 'function') {
                            navigateToState(stateCode);
                        } else {
                            console.error('JS Error: navigateToState function not found. Attempting fallback.');
                            window.location.search = `state=${stateCode}`; // Fallback
                        }
                    }
                });
            }
        """

        # Add GeoJSON layer - Use 'script' parameter to attach JS
        folium.GeoJson(
            us_states_geojson,
            name="States",
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.features.GeoJsonTooltip(
                fields=['name'], # Just show name on hover
                aliases=['State:'], 
                sticky=True,
                style="background-color: rgba(255,255,255,0.9); color: #333; font-family: sans-serif; font-size: 12px; padding: 5px; border: 1px solid #ccc; border-radius: 3px; box-shadow: 0 0 5px rgba(0,0,0,0.3);" 
            ),
            popup=None, # No popup needed
            script=click_script, # Attach the per-feature JS here
            embed=False # Let streamlit-folium handle embedding
        ).add_to(m)

    else: 
        # --- Detail Map Configuration --- 
        state_info = states_data.get(final_display_code) # Get data for the displayed state
        
        # Add embedded back button HTML (optional, as we have the Streamlit one)
        # back_button_html = """ ... """
        # m.get_root().html.add_child(Element(back_button_html))

        # Add state boundary highlight
        state_feature = next((f for f in us_states_geojson['features'] if f.get('properties', {}).get('code') == final_display_code), None)
        if state_feature:
            folium.GeoJson(state_feature, name="Selected State Boundary", 
                        style_function=lambda x: {'color': dark_purple, 'weight': 4, 'fillOpacity': 0.15, 'fillColor': light_purple}, 
                        interactive=False).add_to(m)

        # Add household markers
        if households:
            marker_cluster = MarkerCluster(name="Homes").add_to(m)
            vis_ev, vis_ac, vis_pv = 0, 0, 0 # Reset counts for this state

            for house in households:
                # (Keep household marker logic as before)
                icon_color, icon_name = "gray", "home"; tooltip_parts = []; dev_icons = ""
                if house['has_ev'] and show_ev: icon_color, icon_name = "blue", "plug"; tooltip_parts.append("EV"); vis_ev += 1; dev_icons += '<i class="fa fa-check-circle" style="color:#3366cc;"></i> EV '
                else: dev_icons += '<i class="fa fa-times-circle" style="color:#ccc;"></i> EV '
                if house['has_ac'] and show_ac:
                    if icon_color=="gray": icon_color, icon_name = "orange", "snowflake-o"
                    tooltip_parts.append("AC"); vis_ac += 1; dev_icons += '<i class="fa fa-check-circle" style="color:#ff9900;"></i> AC '
                else: dev_icons += '<i class="fa fa-times-circle" style="color:#ccc;"></i> AC '
                if house['has_pv'] and show_pv:
                    if icon_color=="gray": icon_color, icon_name = "green", "sun-o"
                    tooltip_parts.append("Solar"); vis_pv += 1; dev_icons += '<i class="fa fa-check-circle" style="color:#66cc66;"></i> Solar'
                else: dev_icons += '<i class="fa fa-times-circle" style="color:#ccc;"></i> Solar'
                tooltip = f"Home {house['id']} ({' + '.join(tooltip_parts)})" if tooltip_parts else f"Home {house['id']}"
                popup_html = f"""<div style='min-width:180px; font-size: 13px;'><h4 style='margin:0 0 8px 0; color:#515D9A;'>Home {house['id']}</h4>{dev_icons}<br><b>Consumption:</b> {house['energy_consumption']} kWh/day</div>"""
                folium.Marker([house['lat'], house['lon']], tooltip=tooltip, popup=folium.Popup(popup_html, max_width=250), icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa')).add_to(marker_cluster)
                
            # Add Info Panel with visible counts for the *filtered* households
            if state_info:
                 info_panel_html = f"""<div style='position: absolute; top: 10px; right: 10px; width: auto; background-color: rgba(255,255,255,0.9); border-radius: 5px; box-shadow: 0 0 8px rgba(0,0,0,0.2); padding: 10px; z-index: 1000; font-size: 13px;'><h4 style='margin:0 0 8px 0; color:#515D9A;'>{state_info.get('name', final_display_code)} (Visible Homes)</h4>Count: {len(households)}<br>EV: {vis_ev}<br>AC: {vis_ac}<br>Solar: {vis_pv}</div>"""
                 m.get_root().html.add_child(Element(info_panel_html))
        else:
            # Show message if no households match filters
            folium.Marker(map_center, tooltip="No homes match filters for this state", icon=folium.Icon(color="gray", icon="info-sign")).add_to(m)

    # --- Add Legend and Display Map --- 
    legend_html = """<div style="position: fixed; bottom: 20px; right: 20px; z-index:1000; background-color:rgba(255,255,255,0.9); padding: 8px 10px; border-radius:5px; border: 1px solid #ccc; font-size:12px;"><b>Legend</b><br><i class="fa fa-map-marker" style="color:blue;"></i> EV Home<br><i class="fa fa-map-marker" style="color:orange;"></i> AC Home<br><i class="fa fa-map-marker" style="color:green;"></i> Solar Home<br><i class="fa fa-map-marker" style="color:gray;"></i> Other Home</div>""" # Removed state summary marker from legend
    m.get_root().html.add_child(Element(legend_html))
    folium.LayerControl().add_to(m) 

    # Display map
    folium_static(m, width=None, height=600)

    # --- Data Table (only on detail view) --- 
    if not show_overview:
        st.markdown("### Filtered Homes Data")
        if households:
            df_houses = pd.DataFrame(households)[['id', 'has_ev', 'has_ac', 'has_pv', 'energy_consumption']] 
            df_houses = df_houses.rename(columns={'id': 'Home ID', 'has_ev': 'EV', 'has_ac': 'AC', 'has_pv': 'PV', 'energy_consumption': 'Consumption (kWh/day)'})
            df_houses[['EV', 'AC', 'PV']] = df_houses[['EV', 'AC', 'PV']].replace({True: '✓', False: ''})
            st.dataframe(df_houses, use_container_width=True, hide_index=True)
            
            @st.cache_data
            def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')
            csv = convert_df_to_csv(df_houses)
            st.download_button("Download Data as CSV", csv, f"{state_info.get('name', final_display_code)}_homes_data.csv", "text/csv", key=f"dl_{final_display_code}")
        else:
            st.info("No homes match the current filters for this state.")

    # Footer
    st.markdown("---")
    st.markdown(f"""<div style='text-align:center; color:{primary_purple};'>Explore device distribution by clicking states or using the selector.</div>""", unsafe_allow_html=True)
