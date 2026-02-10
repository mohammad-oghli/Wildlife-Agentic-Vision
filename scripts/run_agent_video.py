from perception.video import load_video_bytes
from agent.loop import run_video_agent
from agent.memory import load_memory, reset_agent_memory
from agent.visual_video import visualize_from_video
from tools.code_runner import run_code
from pathlib import Path


sample_path = "data/samples/wildlife1.mp4"
video_bytes = load_video_bytes(sample_path)

memory = load_memory()
reset_agent_memory(memory)

run_video_agent(video_bytes)

memory = load_memory()
#output_path = clean_video_for_gemini_code_runner(sample_path)
script_path = Path("scripts/visualize_result.py")
code_text = visualize_from_video(sample_path, memory)
#print(code_text)
run_code(code_text, path=script_path)