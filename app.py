"""
Spotify Animation App — a visualization generator for all time Spotify Wrapped-style animations.
Copyright (C) 2025 Maxwell Bernard

Disclaimer: Spotify is a registered trademark of Spotify AB.
This app is a third-party tool that uses Spotify data and is not affiliated with or endorsed by Spotify

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import io
import tempfile
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from modules.create_bar_animation import (
    create_bar_animation,
    days,
    dpi,
    interp_steps,
    period,
)
from modules.create_bar_plot import plot_final_frame
from modules.data_processing import (
    extract_json_from_zip,
    fetch_and_process_files,
    prepare_df_for_visual_anims,
    prepare_df_for_visual_plots,
)
from modules.normalize_inputs import normalize_inputs
from modules.prepare_visuals import error_logged, image_cache
from modules.supabase_client import supabase

st.set_page_config(
    page_title="Spotify Visual Generator",
    page_icon="🎵",
    layout="wide",
)

# custom CSS to style the app
st.markdown(
    """
    <style>
    
    /* Page layout adjustments */
    .stApp {
        margin-top: -52px !important;
        # background: #ecedeb !important;
        background: #e2e3e1 !important;
        # background: #fafafa !important;

    }
    
    /* Sidebar spacing */
    section[data-testid="stSidebar"] {
        margin-top: -55px !important;
        width: 325px !important;
    }
    
    /* Style the entire sidebar */
    section[data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #1DB954, #1ED760, #6FFEA4) !important;
        padding: 15px !important;
        margin: 0px !important;
        box-shadow: 0 0 12px rgba(0,0,0,0.3) !important;
    }
    
    # /* Video borders */
    # video[controls] {
    #     border: 2px solid #d3d3d3 !important;
    #     border-radius: 10px !important;
    #     box-shadow: 0 2px 6px rgba(0, 0, 0, 0.10) !important;
    # }
    
    /* Header background */
    header {
        # background-color: #fafafa !important;
        background-color: #e2e3e1 !important;

    }
    
    /* Content width limit */
    .block-container {
        max-width: 1300px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    
    /* Form styling */
    div[data-testid="stForm"] {
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Sidebar form button styling */
    section[data-testid="stSidebar"] .stFormSubmitButton > button {
        background-color: #ffffff !important;
        color: black !important;
        font-weight: bold !important;
        margin-top: 10px !important;
    }
    
    section[data-testid="stSidebar"] .stFormSubmitButton > button:hover {
        background-color: #fafafa !important;
        color: green !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def track_event(event_type: str, metadata: dict = None, count: int = 1):
    event = {
        "user_id": "anonymous",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "count": count,
        "metadata": metadata or {},
        "user_event": event_type,
    }
    try:
        supabase.table("user_events").insert([event]).execute()
    except Exception as e:
        error_message = str(e).lower()
        if "rate" in error_message or "limit" in error_message:
            print(f"SUPABASE RATE LIMIT: {e}")
        else:
            print(f"SUPABASE ERROR: {e}")


# Initialize session state with defaults
if "form_values" not in st.session_state:
    st.session_state.form_values = {
        "selected_attribute": "artist_name",
        "analysis_metric": "Streams",
        "speed_for_bar_animation": 60,
        "top_n": 5,
        "start_date": datetime(2023, 1, 1),
        "end_date": datetime.now(),
        "data_uploaded": False,
        "data_min_date": None,
        "data_max_date": None,
    }

# Initialize event tracking flags in session state
if "generate_image_clicked" not in st.session_state:
    st.session_state.generate_image_clicked = False

if "download_image_clicked" not in st.session_state:
    st.session_state.download_image_clicked = False

if "generate_animation_clicked" not in st.session_state:
    st.session_state.generate_animation_clicked = False

if "download_animation_clicked" not in st.session_state:
    st.session_state.download_animation_clicked = False

if "data_uploaded" not in st.session_state.form_values:
    st.session_state.form_values["data_uploaded"] = False


# Initialise date range in session state
if "data_min_date" not in st.session_state.form_values:
    st.session_state.form_values["data_min_date"] = None
if "data_max_date" not in st.session_state.form_values:
    st.session_state.form_values["data_max_date"] = None

with st.sidebar:
    st.markdown(
        """
    <link href="https://fonts.googleapis.com/css2?family=Fredoka:wght@700&display=swap" rel="stylesheet">
    <h1 style="
        font-family: 'Fredoka', sans-serif;
        font-size: 26px;
        color: #fafafa;
        text-shadow: 0 0 40px #444444;
        text-align: center;
        letter-spacing: 1px;
        margin-bottom: 0.01em;
        margin-top: 1.2em;
        ">Display Options
    </h1>
    """,
        unsafe_allow_html=True,
    )
    with st.form(key="preferences_form"):
        selected_attribute = st.selectbox(
            "What data do you want to see?", ["Artist", "Song", "Album"]
        )

        if selected_attribute == "Artist":
            selected_attribute = "artist_name"
        elif selected_attribute == "Song":
            selected_attribute = "track_name"
        else:
            selected_attribute = "album_name"

        analysis_metric = st.selectbox(
            "Streams or Time Listened? You Decide!",
            ["Number of Streams", "Time Listened"],
        )

        speed_of_visualization = st.selectbox(
            "How fast do you want the animation?", ["Slow", "Medium", "Fast"], index=1
        )

        if speed_of_visualization == "Slow":
            speed_for_bar_animation = 20
        elif speed_of_visualization == "Medium":
            speed_for_bar_animation = 28
        else:
            speed_for_bar_animation = 36

        top_n = st.slider(
            "How many items do you want to Display?",
            min_value=1,
            max_value=10,
            value=5,
        )

        col1, col2 = st.columns(2)
        with col1:
            min_date = (
                st.session_state.form_values["data_min_date"]
                if st.session_state.form_values["data_uploaded"]
                else st.session_state.form_values["start_date"]
            )
            start_date = st.date_input(
                "Select start date",
                value=st.session_state.form_values["start_date"],
                min_value=min_date,
                max_value=st.session_state.form_values[
                    "end_date"
                ],  # Prevent overlap with end_date
                key="start_date_input",
            )
            start_date = pd.to_datetime(start_date)
        with col2:
            max_date = (
                st.session_state.form_values["data_max_date"]
                if st.session_state.form_values["data_uploaded"]
                else st.session_state.form_values["end_date"]
            )
            end_date = st.date_input(
                "Select end date",
                value=st.session_state.form_values["end_date"],
                min_value=st.session_state.form_values[
                    "start_date"
                ],  # Prevent overlap with start_date
                max_value=max_date,
                key="end_date_input",
            )
            end_date = pd.to_datetime(end_date)

        submit_form = st.form_submit_button(
            label="Apply Preferences",
            use_container_width=True,
        )

    if submit_form:
        st.session_state.form_values.update(
            {
                "selected_attribute": selected_attribute,
                "analysis_metric": analysis_metric,
                "speed_for_bar_animation": speed_for_bar_animation,
                "top_n": top_n,
                "start_date": start_date,
                "end_date": end_date,
            }
        )

# normalise user inputs
selected_attribute, analysis_metric = normalize_inputs(
    selected_attribute, analysis_metric
)

col1, col2, col3 = st.columns(3)
col2.image("2024 Spotify Brand Assets/spotify_full_logo_black.svg", width=235)

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Fredoka:wght@700&display=swap" rel="stylesheet">
    <h1 style='text-align: center;
               font-family: "Fredoka", sans-serif;
               font-size: 48px;
               background: linear-gradient(90deg, #1DB954, #1ED760, #00FFA3, #1DB954);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               letter-spacing: 1.5px;
               margin-bottom: 0.3em;
               background-size: 300% 300%;'>
        Spotify Visual Generator
    </h1>
    <style>
        @keyframes gradient-shift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="text-align: center;">
        <a href="https://github.com/maxwellbernard" target="_blank" style="font-size: 18px;
          font-weight: bold; color: #1db954;">
            Visit my GitHub for more projects and code! 🚀
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="text-align: center; font-size: 16px;">
        Connect with me on <a href="https://linkedin.com/in/maxwell-bernard" target="_blank" style="color: #0a66c2; text-decoration: underline;">LinkedIn</a> 🔗
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center;">
        <p><strong>🎵 Upload your own Spotify data to create visualizations of your entire
          listening history! 🎵</strong></p>
    </div>
""",
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        """
        <div style="border: 1px solid #d3d3d3; padding: 10px; border-radius: 10px;
                    text-align: center; height: 75px;
                    background: linear-gradient(to right, #f5f5f5, #fafafa);
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);">
            <p style="font-size: 16px; font-weight: bold;">1. Upload Your Own Data &nbsp;📥</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div style="border: 1px solid #d3d3d3; padding: 10px; border-radius: 10px;
                    text-align: center; height: 75px;
                    background: linear-gradient(to right, #f5f5f5, #fafafa);
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);">
            <p style="font-size: 16px; font-weight: bold;">2. Customize Your Preferences &nbsp;📊</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div style="border: 1px solid #d3d3d3; padding: 10px; border-radius: 10px;
                    text-align: center; height: 75px;
                    background: linear-gradient(to right, #f5f5f5, #fafafa);
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);">
            <p style="font-size: 16px; font-weight: bold;">3. Generate Animations &nbsp;🎞️</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """
        <div style="border: 1px solid #d3d3d3; padding: 10px; border-radius: 10px;
                    text-align: center; height: 75px;
                    background: linear-gradient(to right, #f5f5f5, #fafafa);
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);">
            <p style="font-size: 16px; font-weight: bold;">4. Download and Share! &nbsp;💾</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("Click here to see what my visualizations look like!"):
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "<h5 style='text-align: center;'>Bar Chart Race &nbsp;🎥</h4>",
            unsafe_allow_html=True,
        )
        # st.video("artist_name_Streams_animation (7).mp4")
        st.video("./visuals/artist_name_Streams_animation_final.mp4")

    with col2:
        st.markdown(
            "<h5 style='text-align: center;'>Bar Chart Image &nbsp;📸</h4>",
            unsafe_allow_html=True,
        )
        st.image("./visuals/album_name_Streams_visual_max.jpg")

    st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Upload Your Data", divider="green")

with st.expander(
    "Click here for Data Upload Guide &nbsp; 📥",
):
    st.markdown(
        "In order to get your extended streaming history files, you must request your data from Spotify."
    )
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([0.5, 0.5])

    with col1:
        st.markdown(
            """
            **Step 1:** &nbsp; Open the [Privacy page](https://www.spotify.com/us/account/privacy/) on the Spotify website.
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "**Step 2:** &nbsp; Scroll down to the 'Download your data' section."
        )
        st.markdown(
            "**Step 3:** &nbsp; Untick &nbsp;❌&nbsp; the Account Data section and tick &nbsp;✅&nbsp; <br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the 'Extended Streaming History' section <br>",
            unsafe_allow_html=True,
        )
        st.markdown("**Step 4:** &nbsp; Click the 'Request data' button.")
        st.markdown(
            "**Step 5:** &nbsp; Spotify will then send you an email to confirm <br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;your request, which you need to confirm.",
            unsafe_allow_html=True,
        )
        st.markdown(
            "**Step 6:** &nbsp; After a few days, your data will be emailed.<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Download the Zip file and upload it below!",
            unsafe_allow_html=True,
        )

    with col2:
        st.image("./visuals/download_guide.png", width=450)


uploaded_file = st.file_uploader(
    "Upload your Spotify data (ZIP File)", type=["zip"], accept_multiple_files=False
)
if uploaded_file and not st.session_state.form_values["data_uploaded"]:
    try:
        json_contents = extract_json_from_zip(uploaded_file)

        if not json_contents:
            st.error(
                "No Streaming History JSON files found in the ZIP file. Please make sure you uploaded the correct ZIP file from Spotify."
            )
        else:
            df = fetch_and_process_files(json_contents)

            start_date_file = df["Date"].min()
            end_date_file = df["Date"].max()

            # Store full data range and update session state
            st.session_state.form_values.update(
                {
                    "start_date": start_date_file,
                    "end_date": end_date_file,
                    "data_min_date": start_date_file,
                    "data_max_date": end_date_file,
                    "data_uploaded": True,
                }
            )

            st.session_state.df = df
            st.rerun()

    except Exception as e:
        st.error(f"Error processing ZIP file: {str(e)}")

elif uploaded_file:
    try:
        json_contents = extract_json_from_zip(uploaded_file)
        if json_contents:
            df = fetch_and_process_files(json_contents)
            st.success("Data uploaded successfully! 🎉")
        else:
            st.error("No valid JSON files found in ZIP.")
    except Exception as e:
        st.error(f"Error processing ZIP file: {str(e)}")
        df = None
else:
    st.warning("Please upload your Spotify ZIP file to proceed.")
    df = None

selected_attribute, analysis_metric = normalize_inputs(
    st.session_state.form_values["selected_attribute"],
    st.session_state.form_values["analysis_metric"],
)


# Initialize session state for storing image bytes
if "bar_plot_bytes" not in st.session_state:
    st.session_state.bar_plot_bytes = None
if "file_name_for_download" not in st.session_state:
    st.session_state.file_name_for_download = None

st.subheader("Generate Image", divider="green")

if st.button("Generate Image", key="generate_images_button"):
    st.session_state.generate_image_clicked = True

if st.session_state.generate_image_clicked:
    track_event(
        "generate_image",
        metadata={
            "selected_attribute": selected_attribute,
            "analysis_metric": analysis_metric,
            "top_n": top_n,
        },
    )
    st.session_state.generate_image_clicked = False

    if uploaded_file:
        with st.spinner("Generating visual..."):
            df_plot = prepare_df_for_visual_plots(
                df,
                selected_attribute=selected_attribute,
                analysis_metric=analysis_metric,
                start_date=start_date,
                end_date=end_date,
                top_n=top_n,
            )

            plt.close("all")

            fig = plot_final_frame(
                df=df_plot,
                top_n=top_n,
                analysis_metric=analysis_metric,
                selected_attribute=selected_attribute,
                start_date=start_date,
                end_date=end_date,
                period=period,
                days=days,
                image_cache=image_cache,
                error_logged=error_logged,
            )

            buf = io.BytesIO()
            fig.savefig(
                buf, format="jpeg", dpi=300, facecolor="#F0F0F0", edgecolor="none"
            )
            buf.seek(0)

            st.session_state.bar_plot_bytes = buf.getvalue()
            st.session_state.file_name_for_download = (
                f"{selected_attribute}_{analysis_metric}_visual.jpg"
            )
    else:
        st.warning("Please upload your Spotify JSON files to proceed.")

if st.session_state.bar_plot_bytes:
    st.markdown(
        "<h4 style='text-align: left;'>Bar Chart Image 📸</h4>",
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([0.55, 0.44, 0.01])

    with col1:
        st.image(st.session_state.bar_plot_bytes)

    with col2:
        col_dl_1, col_dl_2, col_dl_3 = st.columns([0.005, 0.99, 0.005])
        with col_dl_2:
            st.write("Click the button below to download your visual:")
            clicked = st.download_button(
                label="Download Visual",
                data=st.session_state.bar_plot_bytes,
                file_name=st.session_state.file_name_for_download,
                mime="image/jpeg",
                key="download_bar_plot",
            )
            if clicked:
                st.session_state.download_image_clicked = True

            st.markdown(
                "<p style='text-align: left; font-size: 14px; color: gray;'>Then you can upload it to social media and show friends your superior music taste!</p>",
                unsafe_allow_html=True,
            )

if st.session_state.download_image_clicked:
    track_event(
        "download_image",
        metadata={
            "selected_attribute": selected_attribute,
            "analysis_metric": analysis_metric,
            "top_n": top_n,
        },
    )
    st.session_state.download_image_clicked = False


class AnimationState:
    def __init__(self, top_n):
        self.prev_interp_positions = self.prev_positions = (
            self.current_new_positions
        ) = list(range(9, 9 - top_n, -1))
        self.prev_names = [""] * top_n
        self.prev_widths = [0] * top_n


animation_state = AnimationState(top_n)

if "temp_file_path_bar_anim" not in st.session_state:
    st.session_state.temp_file_path_bar_anim = None  # Initialize state

st.subheader("Generate Animation", divider="green")
if st.button("Generate Animation", key="generate_animation_button"):
    st.session_state.generate_animation_clicked = True

if st.session_state.generate_animation_clicked:
    track_event(
        "generate_animation",
        metadata={
            "selected_attribute": selected_attribute,
            "analysis_metric": analysis_metric,
            "top_n": top_n,
        },
    )
    st.session_state.generate_animation_clicked = False  # reset flag

    if uploaded_file:
        with st.spinner("Generating animation..."):
            message_placeholder = st.empty()
            message_placeholder.write(
                "Hold tight, this may take a few minutes if your data covers many years 😬"
            )
            df_anim = prepare_df_for_visual_anims(
                df,
                selected_attribute=selected_attribute,
                analysis_metric=analysis_metric,
                start_date=start_date,
                end_date=end_date,
                top_n=top_n,
            )

            anim_bar_plot = create_bar_animation(
                df_anim,
                top_n,
                analysis_metric,
                selected_attribute,
                period,
                dpi,
                days,
                interp_steps,
                start_date,
                end_date,
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file_path = temp_file.name
                anim_bar_plot.save(
                    temp_file_path,
                    writer="ffmpeg",
                    fps=speed_for_bar_animation,
                    savefig_kwargs={"facecolor": "#F0F0F0"},
                )
                st.session_state.temp_file_path_bar_anim = temp_file_path
    else:
        st.warning("Please upload your Spotify JSON files to proceed.")

if st.session_state.get("temp_file_path_bar_anim"):
    st.markdown(
        "<h4 style='text-align: left;'>Bar Chart Race 📊</h4>",
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([0.55, 0.44, 0.01])
    with col1:
        st.video(st.session_state.temp_file_path_bar_anim)

    with col2:
        col1, col2, col3 = st.columns([0.005, 0.99, 0.005])
        with col2:
            st.write("Click the button below to download your Animation:")
            with open(st.session_state.temp_file_path_bar_anim, "rb") as f:
                clicked = st.download_button(
                    label="Download Animation",
                    data=f.read(),
                    file_name=f"{selected_attribute}_{analysis_metric}_animation.mp4",
                    mime="video/mp4",
                    key="download_bar_animation",
                )
                if clicked:
                    st.session_state.download_animation_clicked = True

if st.session_state.download_animation_clicked:
    track_event(
        "download_animation",
        metadata={
            "selected_attribute": selected_attribute,
            "analysis_metric": analysis_metric,
            "top_n": top_n,
        },
    )
    st.session_state.download_animation_clicked = False  # reset flag

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("License", divider="green")
st.markdown(
    """
    This application is open-source and available under the GNU General Public License v3.

    You are free to use, modify, and distribute this software under the terms of the GPLv3.

    For more details, visit the [GitHub repository](https://github.com/maxwellbernard/spotify_animation_app). 
    """,
    unsafe_allow_html=True,
)

# feedback button
st.markdown(
    """
    <style>
    .feedback-btn {
        font-size: 14px; 
        color: #222 !important; 
        text-decoration: none !important;
        background-color: #f0f0f0; 
        padding: 10px 25px; 
        border-radius: 5px;
        transition: color 0.2s;
        display: inline-block;
        border: 1px solid #ccc;
    }
    .feedback-btn:hover {
        color: #1DB954 !important;
        border: 1px solid #1DB954;
    }
    </style>
    <div style="text-align: center; margin-top: 50px;">
        <a href="https://docs.google.com/forms/d/e/1FAIpQLSepkDYlPQ4AcXyAMPSAYtUcktg0ui2GSsKKL5i58asGbVUO9w/viewform?usp=header" target="_blank" 
           class="feedback-btn">
            Give Feedback &nbsp;✍️
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

footer = """
<style>
.footer {
    width: 100%;
    color: #b0b0b0;
    text-align: center;
    font-size: 11px;
    padding: 8px 0;
    font-family: Arial, sans-serif;
    margin-top: 70px;
}
</style>
<div class="footer">
    Disclaimer: Spotify is a registered trademark of Spotify AB. This app is a third-party tool that uses Spotify data and is not affiliated with or endorsed by Spotify.
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
