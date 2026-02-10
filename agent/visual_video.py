from google.genai import types
import yaml
import os
import time
import json
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
settings = yaml.safe_load(open("config/settings.yaml"))
OUTPUT_PATH = "data/output"    

def visualize_from_video(video_path, memory):

    CODE_BLOCK = """
import cv2
import os

def run_tracker_from_ai(video_path, ai_seed):

# ai_seed example: {{"timestamp_sec": 5, "center_norm": [0.65, 0.6]}}

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 1. Jump to the starting timestamp
start_frame = int(ai_seed['timestamp_sec'] * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

ret, frame = cap.read()
if not ret:
    print("Could not read frame at timestamp.")
    return

# 2. Convert Normalized AI Coordinates to Pixels
cx = int(ai_seed['center_norm'][0] * width)
cy = int(ai_seed['center_norm'][1] * height)

# Define a default box size (e.g., 10% of screen)
w, h = int(width * 0.12), int(height * 0.15)
bbox = (cx - w//2, cy - h//2, w, h) # (x, y, w, h)

# 3. Initialize the Tracker
# CSRT is accurate; KCF is faster but less robust to occlusion
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, bbox)

print("Tracking started. Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Update the tracker for the new frame
    success, box = tracker.update(frame)

    if success:
        (x, y, w, h) = [int(v) for v in box]
        # Draw the Tracking Box (Blue)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "TRACKING: Animal_3", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        cv2.putText(frame, "LOST TARGET", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("AI-Seeded Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Example Usage
seed = {{"timestamp_sec": 10, "center_norm": [0.65, 0.6]}}
run_tracker_from_ai("data/samples/wildlife2.mp4", seed)

"""
    
    
    SYSTEM_PROMPT = f"""
You are a visualization agent.

You are given:
- Behavioral memory data
- A local wildlife video file path

Here is the behavioral memory data you must use:
MEMORY_DATA = {json.dumps(memory, indent=2)}

Local video path:
{video_path}

Your task:
- Generate a complete standalone Python script using OpenCV (cv2)
- The script MUST be directly executable without modification

Core behavior:
- Identify animals marked as "at_risk" in the behavioral memory
- For EACH at-risk animal, use the memory data ONLY as a seed for tracking

Tracking strategy (MANDATORY):
1. Open the video using cv2.VideoCapture
2. Read FPS, width, and height from the video
3. Seed as Absolute Origin
    - The seed timestamp is the definitive start of perception.
    - Frames prior to the seed timestamp MUST NOT be processed,
    analyzed, or used for tracker initialization.
    - All tracking logic begins strictly at the seed frame.
4. Seek to the frames nearest to the time window range provided in memory data for seed attribute [time_window]
5. Convert normalized coordinates from memory data into pixel coordinates using seed attribute [avg_coordinate]
6. Initialize area of interest large bounding box centered on that position 
7. Initialize an OpenCV CSRT tracker using that bounding box
8. Use the tracker to update and draw the bounding box for all subsequent frames
9. Do NOT re-compute or re-invent bounding boxes after initialization
10. Use this Python tracker code as reference:
        {CODE_BLOCK}

Drawing rules:
- Add title [At Risk: class attribute] with blue font 
- Draw ONLY bounding boxes for animals marked "at_risk"
- Bounding boxes must be red (BGR: 0,0,255)
- Label each box with:
  individual_id and confidence
- If the tracker fails, print "LOST TARGET!" 

Output behavior:
- Display the annotated video using cv2.imshow for only seed [time_window]
- Save the output video to:
  {OUTPUT_PATH}/annotated_wildlife.mp4
- Save 3 random images from seed [time_window] with tracker to:
  {OUTPUT_PATH}/tracker_sample_frame
- Use a Linux-compatible codec (mp4v or avc1)

Rules:
- Do NOT invent bounding boxes
- Do NOT include explanations
- Do NOT include markdown
- Do NOT include comments outside the code
- Output ONLY valid Python code

The generated script must be ready to run.
"""
    response = client.models.generate_content(
    model=settings["model"],
    contents=SYSTEM_PROMPT,
    # config=types.GenerateContentConfig(
    #     thinking_config=types.ThinkingConfig(thinking_level="high")
    # ),
    ) 
    return response.text