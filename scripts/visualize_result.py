import cv2
import os
import random

MEMORY_DATA = {
  "individuals": {
    "animal_1": {
      "history": [
        {
          "timestamp": "00:00",
          "individual_id": "animal_1",
          "speed": 0.3,
          "posture_asymmetry": 0.9,
          "distance_from_group": 8.0,
          "group_baseline": {
            "avg_speed": 0.05,
            "avg_distance": 2.5
          },
          "explanation": {
            "seed": {
              "time_window": [
                0.0,
                4.0
              ],
              "avg_coordinate": [
                0.38,
                0.51
              ]
            },
            "description": "Individual is moving with a severe limp, holding the left hind leg completely off the ground."
          }
        },
        {
          "timestamp": "00:05",
          "individual_id": "animal_1",
          "speed": 0.25,
          "posture_asymmetry": 0.9,
          "distance_from_group": 4.0,
          "group_baseline": {
            "avg_speed": 0.05,
            "avg_distance": 2.0
          },
          "explanation": {
            "desc": "Individual continues to hop on three legs towards the group; left hind leg is dangling.",
            "class": "Elk",
            "expected_reason": "Severe injury or fracture to left hind leg"
          }
        },
        {
          "timestamp": "00:10",
          "individual_id": "animal_1",
          "speed": 0.2,
          "posture_asymmetry": 0.8,
          "distance_from_group": 1.0,
          "group_baseline": {
            "avg_speed": 0.1,
            "avg_distance": 1.5
          },
          "explanation": {
            "desc": "Slowing down as it merges with the herd, still avoiding weight bearing on the injured limb.",
            "class": "Elk",
            "expected_reason": "Severe injury or fracture to left hind leg"
          }
        },
        {
          "timestamp": "00:15",
          "individual_id": "animal_1",
          "speed": 0.0,
          "posture_asymmetry": 0.8,
          "distance_from_group": 0.5,
          "group_baseline": {
            "avg_speed": 0.05,
            "avg_distance": 1.5
          },
          "explanation": {
            "desc": "Standing with the group, left hind leg remains lifted in a static posture.",
            "class": "Elk",
            "expected_reason": "Severe injury or fracture to left hind leg"
          }
        }
      ],
      "confidence": 1.0,
      "risk_level": "at_risk"
    },
    "animal_2": {
      "history": [
        {
          "timestamp": "00:00",
          "individual_id": "animal_2",
          "speed": 0.0,
          "posture_asymmetry": 0.0,
          "distance_from_group": 2.0,
          "group_baseline": {
            "avg_speed": 0.05,
            "avg_distance": 2.5
          },
          "explanation": "Standing still, observing surroundings."
        },
        {
          "timestamp": "00:05",
          "individual_id": "animal_2",
          "speed": 0.0,
          "posture_asymmetry": 0.0,
          "distance_from_group": 1.5,
          "group_baseline": {
            "avg_speed": 0.05,
            "avg_distance": 2.0
          },
          "explanation": "Standing stationary as other individuals approach."
        },
        {
          "timestamp": "00:10",
          "individual_id": "animal_2",
          "speed": 0.1,
          "posture_asymmetry": 0.0,
          "distance_from_group": 1.0,
          "group_baseline": {
            "avg_speed": 0.1,
            "avg_distance": 1.5
          },
          "explanation": "Shifting position slightly within the group."
        },
        {
          "timestamp": "00:15",
          "individual_id": "animal_2",
          "speed": 0.0,
          "posture_asymmetry": 0.0,
          "distance_from_group": 0.5,
          "group_baseline": {
            "avg_speed": 0.05,
            "avg_distance": 1.5
          },
          "explanation": "Standing calmly within the cluster."
        }
      ],
      "confidence": 0.30800000000000005,
      "risk_level": "monitor"
    }
  }
}

VIDEO_PATH = "data/samples/wildlife1.mp4"
OUTPUT_VIDEO = "data/output/annotated_wildlife.mp4"
OUTPUT_FRAMES_DIR = "data/output/tracker_sample_frame"

def main():
    if not os.path.exists(os.path.dirname(OUTPUT_VIDEO)):
        os.makedirs(os.path.dirname(OUTPUT_VIDEO))
    if not os.path.exists(OUTPUT_FRAMES_DIR):
        os.makedirs(OUTPUT_FRAMES_DIR)

    target_data = None
    target_id = None
    
    for ind_id, data in MEMORY_DATA["individuals"].items():
        if data.get("risk_level") == "at_risk":
            target_data = data
            target_id = ind_id
            break
    
    if not target_data:
        print("No at_risk animals found in memory.")
        return

    hist_0 = target_data["history"][0]
    seed_info = hist_0.get("explanation", {}).get("seed", {})
    
    if not seed_info:
        print(f"No seed info found for {target_id}")
        return

    time_window = seed_info.get("time_window", [0.0, 5.0])
    avg_coord = seed_info.get("avg_coordinate", [0.5, 0.5])
    
    animal_class = "Unknown"
    for h in target_data["history"]:
        if isinstance(h["explanation"], dict) and "class" in h["explanation"]:
            animal_class = h["explanation"]["class"]
            break

    confidence = target_data.get("confidence", 0.0)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error opening video: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_sec = time_window[0]
    end_sec = time_window[1]
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    ret, frame = cap.read()
    if not ret:
        print("Could not read start frame.")
        cap.release()
        return

    cx = int(avg_coord[0] * width)
    cy = int(avg_coord[1] * height)
    
    w_box = int(width * 0.12)
    h_box = int(height * 0.15)
    bbox = (cx - w_box//2, cy - h_box//2, w_box, h_box)

    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    total_frames_to_process = end_frame - start_frame
    random_indices = sorted(random.sample(range(0, total_frames_to_process), 3))
    
    current_relative_frame = 0
    
    print(f"Tracking initialized for {target_id} (Class: {animal_class}) from {start_sec}s to {end_sec}s.")

    # Process first frame (seed frame)
    success = True
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    
    cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
    
    text_label = f"{target_id} conf:{confidence}"
    cv2.putText(frame, text_label, (p1[0], p1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.putText(frame, f"At Risk: {animal_class}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    out_vid.write(frame)
    cv2.imshow("AI-Seeded Tracker", frame)
    if current_relative_frame in random_indices:
        fpath = os.path.join(OUTPUT_FRAMES_DIR, f"frame_{current_relative_frame}.jpg")
        cv2.imwrite(fpath, frame)
    
    current_relative_frame += 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if current_frame_num > end_frame:
            break

        success, box = tracker.update(frame)

        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f"{target_id} conf:{confidence}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "LOST TARGET!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.putText(frame, f"At Risk: {animal_class}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        out_vid.write(frame)
        cv2.imshow("AI-Seeded Tracker", frame)

        if current_relative_frame in random_indices:
            fpath = os.path.join(OUTPUT_FRAMES_DIR, f"frame_{current_relative_frame}.jpg")
            cv2.imwrite(fpath, frame)
        
        current_relative_frame += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_vid.release()
    cv2.destroyAllWindows()
    print("Processing complete.")

if __name__ == "__main__":
    main()