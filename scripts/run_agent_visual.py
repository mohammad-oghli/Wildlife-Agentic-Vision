from agent.memory import load_memory, reset_agent_memory
from agent.visual_video import visualize_from_video
from tools.code_runner import run_code
from pathlib import Path

memory = load_memory()

if "session" not in memory or "video_path" not in memory["session"]:
    raise RuntimeError("No active session found. Run video agent first.")

sample_path = memory["session"]["video_path"]
script_path = Path("scripts/agent_output_code/visualize_result.py")

print("Running Behavioral Risk Visualization Agent...")
code_text = visualize_from_video(sample_path, memory)
#print(code_text)
run_code(code_text, path=script_path)