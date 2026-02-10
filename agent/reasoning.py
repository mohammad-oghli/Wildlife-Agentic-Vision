from google import genai
import json
import yaml
import os

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set")

client = genai.Client(api_key=api_key)
settings = yaml.safe_load(open("config/settings.yaml"))

SYSTEM_PROMPT = """
You are a wildlife monitoring agent.
You assess behavioral deviations over time.

Rules:
- Compare individuals to group norms
- Require persistence before concern
- Avoid medical or diagnostic claims
- Output JSON only
"""

def reason(observation, memory):
    prompt = f"""
Observation:
{json.dumps(observation, indent=2)}

Historical Memory:
{json.dumps(memory, indent=2)}

Return JSON with:
- deviation_score (0 to 1)
- explanation
"""

    response = client.models.generate_content(
        model=settings["model"],
        contents=[SYSTEM_PROMPT, prompt],
        config={
            'response_mime_type': 'application/json',
        }
    )

    return json.loads(response.text)
