import json
from datetime import datetime
from pathlib import Path

MEMORY_FILE = Path("data/memory.json")

def set_session_metadata(memory: dict, video_path: str, frame_sampling):
    memory["session"] = {
        "video_path": video_path,
        "created_at": datetime.now().date().isoformat(),
        "frame_sampling": frame_sampling
    }
    save_memory(memory)

def load_memory():
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text())
    return {
        "individuals": {}
    }

def save_memory(memory):
    MEMORY_FILE.write_text(json.dumps(memory, indent=2))


def reset_agent_memory(memory):
    # This resets the variable in-place
    memory.clear()
    memory["individuals"] = {}
    save_memory(memory)
    print("Agent memory reset. Ready for next video.")


def update_individual(memory, individual_id, signals):
    individual = memory["individuals"].setdefault(individual_id, {
        "history": [],
        "confidence": 0.0,
        "risk_level": "normal"
    })
    individual["history"].append(signals)
    return memory
