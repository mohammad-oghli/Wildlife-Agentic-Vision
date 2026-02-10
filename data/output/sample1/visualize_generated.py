import cv2
import os
import random
import numpy as np

def main():
    # Behavioral Memory Data
    MEMORY_DATA = {
      "individuals": {
        "animal_1": {
          "history": [
            {
              "timestamp": "00:00",
              "individual_id": "animal_1",
              "speed": 0.1,
              "posture_asymmetry": 0.0,
              "distance_from_group": 0.0,
              "group_baseline": {
                "avg_speed": 0.1,
                "avg_distance": 1.5
              },
              "explanation": "Adult doe standing in river stream, oriented right."
            },
            {
              "timestamp": "00:08",
              "individual_id": "animal_1",
              "speed": 0.2,
              "posture_asymmetry": 0.0,
              "distance_from_group": 0.0,
              "group_baseline": {
                "avg_speed": 0.2,
                "avg_distance": 2.0
              },
              "explanation": "Adult doe walking slowly towards the river bank."
            },
            {
              "timestamp": "00:16",
              "individual_id": "animal_1",
              "speed": 0.1,
              "posture_asymmetry": 0.0,
              "distance_from_group": 0.0,
              "group_baseline": {
                "avg_speed": 0.1,
                "avg_distance": 2.5
              },
              "explanation": "Adult standing on the bank observing."
            }
          ],
          "confidence": 0.52,
          "risk_level": "monitor"
        },
        "animal_2": {
          "history": [
            {
              "timestamp": "00:00",
              "individual_id": "animal_2",
              "speed": 0.1,
              "posture_asymmetry": 0.1,
              "distance_from_group": 2.5,
              "group_baseline": {
                "avg_speed": 0.1,
                "avg_distance": 1.5
              },
              "explanation": "Fawn on the left, standing in water."
            },
            {
              "timestamp": "00:08",
              "individual_id": "animal_2",
              "speed": 0.2,
              "posture_asymmetry": 0.4,
              "distance_from_group": 3.0,
              "group_baseline": {
                "avg_speed": 0.2,
                "avg_distance": 2.0
              },
              "explanation": "Fawn following the group, gait appears slightly uneven as it traverses water."
            },
            {
              "timestamp": "00:16",
              "individual_id": "animal_2",
              "speed": 0.1,
              "posture_asymmetry": 0.9,
              "distance_from_group": 4.0,
              "group_baseline": {
                "avg_speed": 0.1,
                "avg_distance": 2.5
              },
              "explanation": {
                "seed": {
                  "time_window": [
                    15,
                    19
                  ],
                  "avg_coordinate": [
                    0.75,
                    0.6
                  ]
                }
              }
            },
            {
              "timestamp": "00:25",
              "individual_id": "animal_2",
              "speed": 0.1,
              "posture_asymmetry": 0.9,
              "distance_from_group": 0.0,
              "group_baseline": {
                "avg_speed": 0.1,
                "avg_distance": 0.0
              },
              "explanation": {
                "desc": "Individual is isolated, moving slowly with a severe limp, avoiding weight on the front left leg.",
                "class": "White-tailed Deer Fawn",
                "expected_reason": "Physical injury to the front left leg causing mobility impairment."
              }
            }
          ],
          "confidence": 0.8760000000000001,
          "risk_level": "at_risk"
        },
        "animal_3": {
          "history": [
            {
              "timestamp": "00:00",
              "individual_id": "animal_3",
              "speed": 0.1,
              "posture_asymmetry": 0.0,
              "distance_from_group": 1.0,
              "group_baseline": {
                "avg_speed": 0.1,
                "avg_distance": 1.5
              },
              "explanation": "Fawn on the right, standing near adult."
            },
            {
              "timestamp": "00:08",
              "individual_id": "animal_3",
              "speed": 0.2,
              "posture_asymmetry": 0.0,
              "distance_from_group": 1.0,
              "group_baseline": {
                "avg_speed": 0.2,
                "avg_distance": 2.0
              },
              "explanation": "Fawn walking normally alongside the adult."
            },
            {
              "timestamp": "00:16",
              "individual_id": "animal_3",
              "speed": 0.1,
              "posture_asymmetry": 0.0,
              "distance_from_group": 1.0,
              "group_baseline": {
                "avg_speed": 0.1,
                "avg_distance": 2.5
              },
              "explanation": "Healthy fawn standing on the bank."
            }
          ],
          "confidence": 0.28,
          "risk_level": "monitor"
        }
      }
    }

    video_path = "data/samples/wildlife2.mp4"
    output_dir = "data/output"
    images_dir = os.path.join(output_dir, "tracker_sample_frame")
    video_out_path = os.path.join(output_dir, "annotated_wildlife.mp4")

    # Ensure output directories exist
    os.makedirs(images_dir, exist_ok=True)

    # 1. Identify At-Risk Animals and Extract Seed Data
    target_data = None
    target_id = None
    
    for ind_id, ind_data in MEMORY_DATA["individuals"].items():
        if ind_data.get("risk_level") == "at_risk":
            target_id = ind_id
            confidence_val = ind_data.get("confidence", 0.0)
            
            # Search history for seed and class info
            seed_info = None
            class_info = "Unknown"
            
            for entry in ind_data["history"]:
                expl = entry.get("explanation")
                if isinstance(expl, dict):
                    if "seed" in expl:
                        seed_info = expl["seed"]
                    if "class" in expl:
                        class_info = expl["class"]
            
            if seed_info:
                target_data = {
                    "id": target_id,
                    "confidence": f"{confidence_val * 100:.1f}%",
                    "seed": seed_info,
                    "class": class_info
                }
                break
    
    if not target_data:
        print("No at_risk animal with seed data found.")
        return

    # 2. Setup Video Capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. Seed Logic
    time_window = target_data["seed"]["time_window"]
    start_time_sec = time_window[0]
    end_time_sec = time_window[1]
    
    start_frame = int(start_time_sec * fps)
    end_frame = int(end_time_sec * fps)
    
    # Cap end frame to video length
    if end_frame >= total_frames:
        end_frame = total_frames - 1

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Read the seed frame
    ret, frame = cap.read()
    if not ret:
        print("Could not read seed frame.")
        cap.release()
        return

    # 4. Initialize Tracker
    avg_coord = target_data["seed"]["avg_coordinate"] # [x_norm, y_norm]
    cx = int(avg_coord[0] * width)
    cy = int(avg_coord[1] * height)
    
    # Bounding box size (12% width, 15% height)
    bw, bh = int(width * 0.12), int(height * 0.15)
    bbox = (cx - bw//2, cy - bh//2, bw, bh)
    
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    # Output Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

    processed_frames_buffer = []
    
    print(f"Tracking started for {target_data['id']} from {start_time_sec}s to {end_time_sec}s.")

    current_frame_idx = start_frame

    while current_frame_idx <= end_frame:
        # Tracker update logic happens on the current 'frame' loaded.
        # Note: Tracker was initialized on start_frame. We need to process start_frame first for drawing, 
        # but tracker.update is usually for the NEXT frame.
        # However, to be consistent with drawing on the seed frame, we usually draw the init bbox.
        # To simplify the loop, we will process the sequence: 
        # 1. Update tracker (for frames > start_frame)
        # 2. Draw
        # 3. Write/Show
        # 4. Read next
        
        # If it is the very first frame (seed frame), we don't update, we just draw the init bbox.
        if current_frame_idx == start_frame:
            success = True
            box = bbox
        else:
            success, box = tracker.update(frame)

        # Draw UI
        title_text = f"At Risk: {target_data['class']}"
        cv2.putText(frame, title_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        if success:
            (x, y, w, h) = [int(v) for v in box]
            # Red bounding box (BGR: 0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            label = f"{target_data['id']} {target_data['confidence']}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "LOST TARGET!", (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Show and Write
        cv2.imshow("Annotated Output", frame)
        out.write(frame)
        
        # Store for random sampling
        processed_frames_buffer.append(frame.copy())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Prepare for next iteration
        ret, frame = cap.read()
        if not ret:
            break
        current_frame_idx += 1

    # Clean up video resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save 3 random sample images
    if len(processed_frames_buffer) > 0:
        num_samples = min(3, len(processed_frames_buffer))
        random_indices = random.sample(range(len(processed_frames_buffer)), num_samples)
        
        for i, idx in enumerate(random_indices):
            sample_path = os.path.join(images_dir, f"sample_{i+1}.jpg")
            cv2.imwrite(sample_path, processed_frames_buffer[idx])
            print(f"Saved sample image to: {sample_path}")

    print(f"Output video saved to: {video_out_path}")

if __name__ == "__main__":
    main()