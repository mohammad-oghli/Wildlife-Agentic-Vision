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
          "speed": 0.1,
          "posture_asymmetry": 0.9,
          "distance_from_group": 5.0,
          "group_baseline": {
            "avg_speed": 0.1,
            "avg_distance": 2.0
          },
          "explanation": {
            "seed": {
              "time_window": [
                0.0,
                6.0
              ],
              "avg_coordinate": [
                0.35,
                0.48
              ]
            }
          }
        },
        {
          "timestamp": "00:05",
          "individual_id": "animal_1",
          "speed": 0.5,
          "posture_asymmetry": 1.0,
          "distance_from_group": 3.0,
          "group_baseline": {
            "avg_speed": 0.3,
            "avg_distance": 3.0
          },
          "explanation": {
            "desc": "Severe limp, rear left leg dangling loosely while hopping.",
            "class": "Elk",
            "expected_reason": "Severe leg injury or fracture"
          }
        },
        {
          "timestamp": "00:10",
          "individual_id": "animal_1",
          "speed": 0.4,
          "posture_asymmetry": 1.0,
          "distance_from_group": 1.0,
          "group_baseline": {
            "avg_speed": 0.3,
            "avg_distance": 2.5
          },
          "explanation": {
            "desc": "Continuing to hobble on three legs, keeping pace with group.",
            "class": "Elk",
            "expected_reason": "Severe leg injury or fracture"
          }
        },
        {
          "timestamp": "00:15",
          "individual_id": "animal_1",
          "speed": 0.2,
          "posture_asymmetry": 0.9,
          "distance_from_group": 2.0,
          "group_baseline": {
            "avg_speed": 0.2,
            "avg_distance": 2.0
          },
          "explanation": {
            "desc": "Slowing down, injury still visibly affecting gait.",
            "class": "Elk",
            "expected_reason": "Severe leg injury or fracture"
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
          "distance_from_group": 1.0,
          "group_baseline": {
            "avg_speed": 0.1,
            "avg_distance": 2.0
          },
          "explanation": "Standing still, observing surroundings."
        },
        {
          "timestamp": "00:05",
          "individual_id": "animal_2",
          "speed": 0.3,
          "posture_asymmetry": 0.0,
          "distance_from_group": 1.5,
          "group_baseline": {
            "avg_speed": 0.3,
            "avg_distance": 3.0
          },
          "explanation": "Moving slowly to the right with the group."
        },
        {
          "timestamp": "00:10",
          "individual_id": "animal_2",
          "speed": 0.3,
          "posture_asymmetry": 0.0,
          "distance_from_group": 1.0,
          "group_baseline": {
            "avg_speed": 0.3,
            "avg_distance": 2.5
          },
          "explanation": "Walking calmly."
        },
        {
          "timestamp": "00:15",
          "individual_id": "animal_2",
          "speed": 0.1,
          "posture_asymmetry": 0.0,
          "distance_from_group": 1.0,
          "group_baseline": {
            "avg_speed": 0.2,
            "avg_distance": 2.0
          },
          "explanation": "Grazing or standing near group center."
        }
      ],
      "confidence": 0.5400000000000001,
      "risk_level": "monitor"
    }
  },
  "session": {
    "video_path": "data/samples/wildlife1.mp4",
    "created_at": "2026-02-11",
    "frame_sampling": 5
  }
}

def main():
    # 1. Setup Directories
    output_video_path = "data/output/annotated_wildlife.mp4"
    tracker_sample_dir = "data/output/tracker_sample_frame"
    
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.makedirs(tracker_sample_dir, exist_ok=True)

    # 2. Identify At-Risk Target and Seed Data
    target_id = None
    target_data = None
    seed_info = None
    animal_class = "Unknown"

    for aid, data in MEMORY_DATA["individuals"].items():
        if data.get("risk_level") == "at_risk":
            target_id = aid
            target_data = data
            
            # Find class attribute in history (look ahead if needed)
            for h in data["history"]:
                if isinstance(h.get("explanation"), dict) and "class" in h["explanation"]:
                    animal_class = h["explanation"]["class"]
                    break
            
            # Find seed in history
            for h in data["history"]:
                if isinstance(h.get("explanation"), dict) and "seed" in h["explanation"]:
                    seed_info = h["explanation"]["seed"]
                    break
            break

    if not target_id or not seed_info:
        print("No at_risk individual or seed data found.")
        return

    video_path = MEMORY_DATA["session"]["video_path"]
    
    # 3. Video Initialization
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 4. Seed Logic (Time and Space)
    time_window = seed_info["time_window"] # [start_sec, end_sec]
    avg_coord = seed_info["avg_coordinate"] # [x_norm, y_norm]
    
    start_frame = int(time_window[0] * fps)
    end_frame = int(time_window[1] * fps)
    
    # Generate random indices for sampling 3 frames
    total_frames_in_window = end_frame - start_frame
    sample_indices = []
    if total_frames_in_window > 0:
        sample_indices = random.sample(range(start_frame, end_frame), min(3, total_frames_in_window))

    # Set Video to Start Frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    
    if not ret:
        print("Could not read start frame.")
        cap.release()
        return

    # 5. Tracker Initialization
    # Convert Normalized AI Coordinates to Pixels
    cx = int(avg_coord[0] * width)
    cy = int(avg_coord[1] * height)

    # Define bounding box (approx 12% width, 15% height as per instructions)
    w_box, h_box = int(width * 0.12), int(height * 0.15)
    bbox = (cx - w_box//2, cy - h_box//2, w_box, h_box)

    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    current_frame_idx = start_frame

    print(f"Tracking {target_id} from frame {start_frame} to {end_frame}...")

    # 6. Processing Loop
    while current_frame_idx < end_frame:
        success, box = tracker.update(frame)

        # Drawing
        if success:
            (x, y, w, h) = [int(v) for v in box]
            # Red box (BGR: 0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Label: ID and Confidence
            label = f"{target_id} ({target_data['confidence']})"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "LOST TARGET!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Title: At Risk: [Class] (Blue)
        title = f"At Risk: {animal_class}"
        cv2.putText(frame, title, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Output
        out.write(frame)
        cv2.imshow("AI-Seeded Tracker", frame)

        # Save Sample Images
        if current_frame_idx in sample_indices:
            sample_path = os.path.join(tracker_sample_dir, f"frame_{current_frame_idx}.jpg")
            cv2.imwrite(sample_path, frame)

        # Next Frame
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame_idx += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete.")

if __name__ == "__main__":
    main()