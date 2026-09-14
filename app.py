import streamlit as st
import yaml
import json
import subprocess
from pathlib import Path
from perception.video import load_video_bytes
from agent.loop import run_video_agent
from agent.memory import load_memory, reset_agent_memory, set_session_metadata
from agent.visual_video import visualize_from_video
from tools.code_runner import run_code

# --------------------------
# Paths
# --------------------------
SAMPLES_DIR = Path("data/samples")
OUTPUT_PATH = Path("data/output")
TRACKER_FRAMES_DIR = Path("data/output/tracker_sample_frame")
MEMORY_FILE = Path("data/memory.json")
#OUTPUT_MEMORY = Path("data/memory.json")
SCRIPT_PATH = Path("scripts/agent_output_code/visualize_result.py")
SETTINGS_PATH = Path("config/settings.yaml")

# --------------------------
# UI
# --------------------------
st.set_page_config(layout="wide")
st.title("👁 Behavioral Risk Vision Agent Dashboard")

# --------------------------
# Load available videos
# --------------------------
video_files = [f.name for f in SAMPLES_DIR.glob("*") if f.suffix in [".mp4", ".avi", ".mov"]]

if not video_files:
    st.error("No videos found in data/samples directory.")
    st.stop()

selected_video = st.selectbox("Select a sample video:", video_files)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.video(str(SAMPLES_DIR / selected_video), width=300)

    # --------------------------
    # Run Agent Button
    # --------------------------
    if st.button("⚡ Run Vision Agent"):

        with st.spinner("Running Behavioral Risk Vision Agent..."):

            settings = yaml.safe_load(open(SETTINGS_PATH))

            sample_path = SAMPLES_DIR / selected_video
            video_bytes = load_video_bytes(sample_path)

            memory = load_memory()
            reset_agent_memory(memory)

            set_session_metadata(
                memory,
                video_path=str(sample_path),
                frame_sampling=settings["frame_sampling_seconds"]
            )

            run_video_agent(video_bytes)

        st.success("Vision Agent completed!")

        # --------------------------
        # Run Visualization Agent
        # --------------------------
        with st.spinner("Running Visualization Agent..."):

            memory = load_memory()
            code_text = visualize_from_video(str(sample_path), memory)
            run_code(code_text, path=SCRIPT_PATH)

        st.success("Visualization Agent completed!")

# --------------------------
# Display Results
# --------------------------

# --------------------------
# OUTPUT VIDEO
# --------------------------

video = Path("data/output/annotated_wildlife.mp4")
web_video = Path("data/output/annotated_wildlife_web.mp4")

subprocess.run([
    "ffmpeg",
    "-y",
    "-i", str(video),
    "-vcodec", "libx264",
    "-acodec", "aac",
    str(web_video)
])

col1, col2, col3 = st.columns([1,2,1])
#st.container()

with col2:
    st.subheader("🎥 Annotated Output Video")

    output_video = OUTPUT_PATH / "annotated_wildlife_web.mp4"

    if output_video.exists():
        video_file = open(output_video, "rb")
        st.video(video_file.read(), width=600)
    else:
        st.info("No output video generated yet.")

# ===============================
# Show Tracker Images
# ===============================
st.subheader("🖼 Tracker Sample Frames")

if TRACKER_FRAMES_DIR.exists():
    images = sorted(TRACKER_FRAMES_DIR.glob("*.jpg"))

    if images:
        cols = st.columns(4)
        for i, img in enumerate(images):
            cols[i % 4].image(str(img), caption=img.name)
    else:
        st.info("No tracker frames found.")
else:
    st.info("tracker_sample_frame directory not found.")

# --------------------------
# MEMORY JSON
# --------------------------
st.subheader("📄 Vision Memory")

with st.expander("+ Show Memory"):
    if MEMORY_FILE.exists():
        memory_data = json.load(open(MEMORY_FILE))
        st.json(memory_data)
    else:
        st.info("Memory file not found yet.")

# --------------------------
# OUTPUT DIRECTORY EXPLORER
# --------------------------
st.subheader("📂 Output Directory")

with st.expander("+ Show Directory Explorer"):

    if OUTPUT_PATH.exists():
        files = list(OUTPUT_PATH.glob("*"))
        if files:
            for file in files:
                file_size = round(file.stat().st_size / 1024, 2)
                st.write(f"📁 {file.name} — {file_size} KB")
        else:
            st.info("Output directory is empty.")
    else:
        st.warning("Output directory does not exist.")


