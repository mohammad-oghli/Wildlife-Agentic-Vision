import yaml
from agent.memory import load_memory, save_memory, update_individual
from agent.reasoning import reason
from agent.reason_video import reason_from_video
from agent.confidence import update_confidence
from tools.escalation import flag_at_risk, monitor

settings = yaml.safe_load(open("config/settings.yaml"))
THRESHOLD = settings["confidence_threshold"]

def run_agent(observation):
    memory = load_memory()
    individual_id = observation["individual_id"]

    memory = update_individual(memory, individual_id, observation)

    result = reason(observation, memory)
    deviation = result["deviation_score"]
    explanation = result["explanation"]

    individual = memory["individuals"][individual_id]
    persistence = len(individual["history"])
    confidence = update_confidence(individual["confidence"], deviation, persistence)

    individual["confidence"] = confidence

    if confidence >= THRESHOLD:
        individual["risk_level"] = "at_risk"
        flag_at_risk(individual_id, confidence, explanation)
    else:
        individual["risk_level"] = "monitor"
        monitor(individual_id)

    save_memory(memory)


def run_video_agent(video_bytes):
    memory = load_memory()

    extraction = reason_from_video(video_bytes)
    observations = extraction["observations"]

    for obs in observations:
        individual_id = obs["individual_id"]
        explanation = obs["explanation"]

        memory = update_individual(memory, individual_id, obs)

        history_len = len(memory["individuals"][individual_id]["history"])
        prev_conf = memory["individuals"][individual_id]["confidence"]

        # Simple deviation heuristic (Gemini already normalized inputs)
        deviation = (
            abs(obs["speed"] - obs["group_baseline"]["avg_speed"]) * 0.4 +
            obs["posture_asymmetry"] * 0.4 +
            abs(obs["distance_from_group"] - obs["group_baseline"]["avg_distance"]) * 0.2
        )

        confidence = update_confidence(prev_conf, deviation, history_len)
        memory["individuals"][individual_id]["confidence"] = confidence

        if confidence >= THRESHOLD:
            memory["individuals"][individual_id]["risk_level"] = "at_risk"
            flag_at_risk(individual_id, confidence, explanation)
        else:
            memory["individuals"][individual_id]["risk_level"] = "monitor"
            monitor(individual_id)

    save_memory(memory)
