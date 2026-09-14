from perception.video import load_video_bytes
from agent.loop import run_video_agent
from agent.memory import load_memory, reset_agent_memory, set_session_metadata
from agent.visual_video import visualize_from_video
from tools.code_runner import run_code
from pathlib import Path
import yaml


settings = yaml.safe_load(open("config/settings.yaml"))
input_video = input("Please Enter Input Video: ")
sample_path = "data/samples/" + input_video
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


MEMORY_FILE = Path("data/vision_memory.json")
memory = load_memory(MEMORY_FILE)

if "session" not in memory or "video_path" not in memory["session"]:
    raise RuntimeError("No active session found. Run video agent first.")

sample_path = memory["session"]["video_path"]
script_path = Path("scripts/agent_output_code/visualize_result.py")

print("Running Behavioral Risk Visualization Agent...")
code_text = visualize_from_video(sample_path, memory)
print("Visualization Script Generated: agent_output_code/visualize_result.py")
#print(code_text)
run_code(code_text, path=script_path)