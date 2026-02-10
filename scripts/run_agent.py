from agent.loop import run_agent
import time
from agent.memory import load_memory, reset_agent_memory

observations_1 = [

    # --- Timestep 1 ---
    {
        "individual_id": "animal_1",
        "speed": 0.95,
        "posture_asymmetry": 0.1,
        "distance_from_group": 4.8,
        "group_baseline": {"avg_speed": 0.9, "avg_distance": 5.1}
    },
    {
        "individual_id": "animal_2",
        "speed": 0.75,
        "posture_asymmetry": 0.2,
        "distance_from_group": 7.5,
        "group_baseline": {"avg_speed": 0.9, "avg_distance": 5.1}
    },
    {
        "individual_id": "animal_4",
        "speed": 0.65,
        "posture_asymmetry": 0.5,
        "distance_from_group": 6.2,
        "group_baseline": {"avg_speed": 0.9, "avg_distance": 5.1}
    },

    # --- Timestep 2 ---
    {
        "individual_id": "animal_1",
        "speed": 0.92,
        "posture_asymmetry": 0.1,
        "distance_from_group": 5.0,
        "group_baseline": {"avg_speed": 0.9, "avg_distance": 5.1}
    },
    {
        "individual_id": "animal_2",
        "speed": 0.6,
        "posture_asymmetry": 0.3,
        "distance_from_group": 10.2,
        "group_baseline": {"avg_speed": 0.9, "avg_distance": 5.1}
    },
    {
        "individual_id": "animal_4",
        "speed": 0.5,
        "posture_asymmetry": 0.65,
        "distance_from_group": 7.8,
        "group_baseline": {"avg_speed": 0.9, "avg_distance": 5.1}
    },

    # --- Timestep 3 ---
    {
        "individual_id": "animal_1",
        "speed": 0.9,
        "posture_asymmetry": 0.1,
        "distance_from_group": 5.3,
        "group_baseline": {"avg_speed": 0.9, "avg_distance": 5.1}
    },
    {
        "individual_id": "animal_2",
        "speed": 0.45,
        "posture_asymmetry": 0.4,
        "distance_from_group": 13.6,
        "group_baseline": {"avg_speed": 0.9, "avg_distance": 5.1}
    },
    {
        "individual_id": "animal_4",
        "speed": 0.4,
        "posture_asymmetry": 0.75,
        "distance_from_group": 9.4,
        "group_baseline": {"avg_speed": 0.9, "avg_distance": 5.1}
    }
]

observations_2 = [

    # --- Timestep 1 ---
    {
        "individual_id": "animal_7",
        "speed": 0.55,
        "posture_asymmetry": 0.6,
        "distance_from_group": 8.4,
        "group_baseline": {"avg_speed": 0.9, "avg_distance": 5.0}
    },
    {
        "individual_id": "animal_8",
        "speed": 0.93,
        "posture_asymmetry": 0.1,
        "distance_from_group": 4.9,
        "group_baseline": {"avg_speed": 0.9, "avg_distance": 5.0}
    },

    # --- Timestep 2 ---
    {
        "individual_id": "animal_7",
        "speed": 0.4,
        "posture_asymmetry": 0.75,
        "distance_from_group": 11.7,
        "group_baseline": {"avg_speed": 0.9, "avg_distance": 5.0}
    },
    {
        "individual_id": "animal_8",
        "speed": 0.91,
        "posture_asymmetry": 0.1,
        "distance_from_group": 5.1,
        "group_baseline": {"avg_speed": 0.9, "avg_distance": 5.0}
    },

    # --- Timestep 3 ---
    {
        "individual_id": "animal_7",
        "speed": 0.3,
        "posture_asymmetry": 0.85,
        "distance_from_group": 15.2,
        "group_baseline": {"avg_speed": 0.9, "avg_distance": 5.0}
    },
    {
        "individual_id": "animal_8",
        "speed": 0.9,
        "posture_asymmetry": 0.1,
        "distance_from_group": 5.0,
        "group_baseline": {"avg_speed": 0.9, "avg_distance": 5.0}
    }
]


memory = load_memory()
reset_agent_memory(memory)

for obs in observations_2:
    run_agent(obs)
    time.sleep(4)  # Ensure you stay under the "Requests Per Minute" limit
    
