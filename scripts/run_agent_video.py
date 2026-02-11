from perception.video import load_video_bytes
from agent.loop import run_video_agent
from agent.memory import load_memory, reset_agent_memory, set_session_metadata
import yaml

settings = yaml.safe_load(open("config/settings.yaml"))
sample_path = "data/samples/wildlife1.mp4"
video_bytes = load_video_bytes(sample_path)

memory = load_memory()
reset_agent_memory(memory)

set_session_metadata(
    memory,
    video_path=sample_path,
    frame_sampling=settings["frame_sampling_seconds"]
)

print("Running Behavioral Risk Vision Agent...")
run_video_agent(video_bytes)