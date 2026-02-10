import cv2
import os
import random

# Behavioral Memory Data
MEMORY_DATA = {
  "individuals": {
    "animal_1": {
      "history": [
        {
          "timestamp": "00:00",
          "individual_id": "animal_1",
          "speed": 0.2,
          "posture_asymmetry": 0.0,
          "distance_from_group": 4.0,
          "group_baseline": {
            "avg_speed": 0.1,
            "avg_distance": 3.0
          },
          "explanation": "Walking slowly to the left of the group"
        },
        {
          "timestamp": "00:06",
          "individual_id": "animal_1",
          "speed": 0.4,
          "posture_asymmetry": 0.1,
          "distance_from_group": 5.0,
          "group_baseline": {
            "avg_speed": 0.4,
            "avg_distance": 3.5
          },
          "explanation": "Walking away from camera"
        },
        {
          "timestamp": "00:12",
          "individual_id": "animal_1",
          "speed": 0.2,
          "posture_asymmetry": 0.0,
          "distance_from_group": 6.0,
          "group_baseline": {
            "avg_speed": 0.2,
            "avg_distance": 2.5
          },
          "explanation": "Slowing down on the left flank"
        }
      ],
      "confidence": 0.6040000000000001,
      "risk_level": "monitor"
    },
    "animal_2": {
      "history": [
        {
          "timestamp": "00:00",
          "individual_id": "animal_2",
          "speed": 0.1,
          "posture_asymmetry": 1.0,
          "distance_from_group": 1.0,
          "group_baseline": {
            "avg_speed": 0.1,
            "avg_distance": 3.0
          },
          "explanation": {
            "seed": {
              "time_window": [
                0.0,
                5.0
              ],
              "avg_coordinate": [
                0.35,
                0.5
              ]
            }
          }
        },
        {
          "timestamp": "00:06",
          "individual_id": "animal_2",
          "speed": 0.3,
          "posture_asymmetry": 0.9,
          "distance_from_group": 2.0,
          "group_baseline": {
            "avg_speed": 0.4,
            "avg_distance": 3.5
          },
          "explanation": {
            "desc": "Severe limp; rear left leg is dangling and non-weight bearing while moving",
            "class": "Elk (Cervus canadensis)",
            "expected_reason": "Traumatic leg injury or fracture"
          }
        },
        {
          "timestamp": "00:12",
          "individual_id": "animal_2",
          "speed": 0.2,
          "posture_asymmetry": 0.9,
          "distance_from_group": 1.5,
          "group_baseline": {
            "avg_speed": 0.2,
            "avg_distance": 2.5
          },
          "explanation": {
            "desc": "Continuing to hop on three legs, merging into the group cluster",
            "class": "Elk (Cervus canadensis)",
            "expected_reason": "Traumatic leg injury or fracture"
          }
        },
        {
          "timestamp": "00:18",
          "individual_id": "animal_2",
          "speed": 0.0,
          "posture_asymmetry": 0.8,
          "distance_from_group": 0.0,
          "group_baseline": {
            "avg_speed": 0.05,
            "avg_distance": 2.0
          },
          "explanation": {
            "desc": "Standing with group, injured leg visibly held off the ground",
            "class": "Elk (Cervus canadensis)",
            "expected_reason": "Traumatic leg injury or fracture"
          }
        }
      ],
      "confidence": 1.0,
      "risk_level": "at_risk"
    },
    "animal_3": {
      "history": [
        {
          "timestamp": "00:00",
          "individual_id": "animal_3",
          "speed": 0.0,
          "posture_asymmetry": 0.0,
          "distance_from_group": 2.0,
          "group_baseline": {
            "avg_speed": 0.1,
            "avg_distance": 3.0
          },
          "explanation": "Standing still to the right of the injured individual"
        },
        {
          "timestamp": "00:06",
          "individual_id": "animal_3",
          "speed": 0.4,
          "posture_asymmetry": 0.0,
          "distance_from_group": 1.0,
          "group_baseline": {
            "avg_speed": 0.4,
            "avg_distance": 3.5
          },
          "explanation": "Moving away from camera in sync with group"
        },
        {
          "timestamp": "00:12",
          "individual_id": "animal_3",
          "speed": 0.2,
          "posture_asymmetry": 0.0,
          "distance_from_group": 0.5,
          "group_baseline": {
            "avg_speed": 0.2,
            "avg_distance": 2.5
          },
          "explanation": "Clustering with other herd members"
        },
        {
          "timestamp": "00:18",
          "individual_id": "animal_3",
          "speed": 0.0,
          "posture_asymmetry": 0.0,
          "distance_from_group": 1.0,
          "group_baseline": {
            "avg_speed": 0.05,
            "avg_distance": 2.0
          },
          "explanation": "Standing near the injured individual"
        }
      ],
      "confidence": 0.664,
      "risk_level": "monitor"
    },
    "animal_4": {
      "history": [
        {
          "timestamp": "00:00",
          "individual_id": "animal_4",
          "speed": 0.0,
          "posture_asymmetry": 0.0,
          "distance_from_group": 5.0,
          "group_baseline": {
            "avg_speed": 0.1,
            "avg_distance": 3.0
          },
          "explanation": "Grazing on the far right periphery"
        },
        {
          "timestamp": "00:06",
          "individual_id": "animal_4",
          "speed": 0.4,
          "posture_asymmetry": 0.0,
          "distance_from_group": 4.0,
          "group_baseline": {
            "avg_speed": 0.4,
            "avg_distance": 3.5
          },
          "explanation": "Joining the main group movement"
        }
      ],
      "confidence": 0.12800000000000003,
      "risk_level": "monitor"
    }
  }
}

VIDEO_PATH = 'data/samples/wildlife1.mp4'
OUTPUT_VIDEO = 'data/output/annotated_wildlife.mp4'
SNAPSHOT_DIR = 'data/output/tracker_sample_frame'

def main():
    # 1. Identify Target and Parse Seed
    target_id = None
    target_data = None
    seed_config = None
    animal_class = "Unknown"

    for ind_id, ind_data in MEMORY_DATA['individuals'].items():
        if ind_data.get('risk_level') == 'at_risk':
            target_id = ind_id
            target_data = ind_data
            
            # Parse history for seed and class info
            for entry in ind_data['history']:
                expl = entry.get('explanation')
                if isinstance(expl, dict):
                    if 'seed' in expl:
                        seed_config = expl['seed']
                    if 'class' in expl:
                        animal_class = expl['class']
            break

    if not target_id or not seed_config:
        print("No at-risk target with seed configuration found.")
        return

    # 2. Setup Video Capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Failed to open video: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. Setup Outputs
    if not os.path.exists(os.path.dirname(OUTPUT_VIDEO)):
        os.makedirs(os.path.dirname(OUTPUT_VIDEO))
    
    if not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # 4. Calculate Seed Frames (Absolute Origin)
    start_time = seed_config['time_window'][0]
    end_time = seed_config['time_window'][1]
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Clamp end frame
    if end_frame >= total_frames:
        end_frame = total_frames - 1

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read start frame.")
        return

    # 5. Initialize Tracker based on Seed
    norm_x, norm_y = seed_config['avg_coordinate']
    cx = int(norm_x * width)
    cy = int(norm_y * height)
    
    # Define bounding box dimensions (approx 12% width, 15% height)
    w_box = int(width * 0.12)
    h_box = int(height * 0.15)
    bbox = (cx - w_box // 2, cy - h_box // 2, w_box, h_box)

    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    # 6. Determine random frames for snapshots
    frame_list = list(range(start_frame, end_frame + 1))
    if len(frame_list) >= 3:
        snapshot_frames = set(random.sample(frame_list, 3))
    else:
        snapshot_frames = set(frame_list)

    print(f"Tracking {target_id} from frame {start_frame} to {end_frame}...")

    # 7. Processing Loop
    current_frame_idx = start_frame

    while current_frame_idx <= end_frame:
        # Note: 'frame' is already read for the current iteration
        
        # Update tracker
        success, box = tracker.update(frame)

        # Draw UI
        if success:
            (x, y, w, h) = [int(v) for v in box]
            # Red Bounding Box (BGR: 0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Label
            label = f"{target_id} (Conf: {target_data['confidence']})"
            cv2.putText(frame, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "LOST TARGET!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Title (Blue: 255, 0, 0)
        title = f"At Risk: {animal_class}"
        cv2.putText(frame, title, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # Save Output Video
        out.write(frame)

        # Save Snapshots
        if current_frame_idx in snapshot_frames:
            snap_path = os.path.join(SNAPSHOT_DIR, f"frame_{current_frame_idx}.jpg")
            cv2.imwrite(snap_path, frame)

        # Display
        cv2.imshow("Annotated Wildlife Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Prepare for next iteration
        ret, frame = cap.read()
        if not ret:
            break
        current_frame_idx += 1

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Script execution finished.")

if __name__ == "__main__":
    main()