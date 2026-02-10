import json
from pathlib import Path

MEMORY_FILE = Path("data/memory.json")

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
