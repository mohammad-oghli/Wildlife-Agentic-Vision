import os
import json
import yaml
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
settings = yaml.safe_load(open("config/settings.yaml"))

SYSTEM_PROMPT = """
You are a wildlife monitoring agent.

You analyze video directly and extract structured behavioral observations.

For EACH visible animal:
- Assign a stable individual_id (animal_1, animal_2, ...)
- Estimate relative speed (0.0–1.0, relative to group)
- Estimate posture asymmetry (0.0–1.0)
- Estimate distance from group center (in meters)
- Estimate group baseline averages

Rules:
- Do NOT diagnose illness or injury
- Use relative, approximate values
- Output ONLY valid JSON
"""

def reason_from_video(video_bytes: bytes, frame_sample=2):
    prompt = """
Analyze each video frame individually every {frame_sample} seconds and return JSON in the following format.
Do not skip any frame.
{
  "observations": [
    {
      "timestamp": "00:00",
      "individual_id": "animal_1",
      "speed": 0.0,
      "posture_asymmetry": 0.0,
      "distance_from_group": 0.0,
      "group_baseline": {
        "avg_speed": 0.0,
        "avg_distance": 0.0
      },
      "explanation": short description
    } 
  ]
}
- if at risk confirmed: 
    1. add to explanation only at the earliest timestamp where it is clearly visible the attribute seed:
       - get approximate time window range in seconds for seed and estimate the average of coordinates for it
        "explanation":{
            seed:
                {
                    time_window: [start, end] 
                    avg coordinate: center coordinates as [x, y] normalized from 0.0 to 1.0 
                }
        }
    2. add to explanation for subsequent timestampes the attributes:
        "explanation":{
            desc: short description
            class: type of animal
            expected reason: possible reason for behavioral deviation
        }
"""

    response = client.models.generate_content(
        model=settings["model"],
        contents = types.Content(
            parts=[
                # System instruction
                types.Part(text=SYSTEM_PROMPT),

                # Task prompt
                types.Part(text=prompt),

                # Video input with resolution control
                types.Part(
                    inline_data=types.Blob(
                        data=video_bytes,
                        mime_type="video/mp4",
                    ),
                    video_metadata=types.VideoMetadata(fps=1),
                    media_resolution=types.PartMediaResolution(
                        level=types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH
                    )
                )
            ]
        ),
        config = types.GenerateContentConfig(
                response_mime_type='application/json',
                # thinking_config=types.ThinkingConfig(
                #     thinking_level='high',
                # )
        )
    )
    return json.loads(response.text)
