import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Read the CSV file containing video paths
csv_file_path = 'path_to_your_csv_file.csv'
video_data = pd.read_csv(csv_file_path)

# Threshold for detecting significant movement
movement_threshold = 0.03  # Adjust this value as needed
sum_threshold = 0.2  # Example threshold for sum of ratios

# Create the output CSV file with the same headers
output_csv_path = 'selected_videos.csv'
video_data.iloc[0:0].to_csv(output_csv_path, index=False)  # Write headers only

# Process each video listed in the CSV file
for idx, row in video_data.iterrows():
    video_path = row[0]  # Assuming the first column contains the video paths

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)

    left_hand_coords = []
    right_hand_coords = []
    total_frames = 0

    prev_left_x, prev_left_y = None, None
    prev_right_x, prev_right_y = None, None

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            total_frames += 1

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    x = hand_landmarks.landmark[8].x
                    y = hand_landmarks.landmark[8].y

                    if label == 'Left':
                        if prev_left_x is None or abs(x - prev_left_x) > movement_threshold or abs(y - prev_left_y) > movement_threshold:
                            left_hand_coords.append([x, y])
                        prev_left_x, prev_left_y = x, y

                    elif label == 'Right':
                        if prev_right_x is None or abs(x - prev_right_x) > movement_threshold or abs(y - prev_right_y) > movement_threshold:
                            right_hand_coords.append([x, y])
                        prev_right_x, prev_right_y = x, y

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()

    left_rows = len(left_hand_coords)
    right_rows = len(right_hand_coords)

    left_hand_ratio = left_rows / total_frames if total_frames > 0 else 0
    right_hand_ratio = right_rows / total_frames if total_frames > 0 else 0

    # Sum of ratios
    ratio_sum = left_hand_ratio + right_hand_ratio

    # Check if sum exceeds threshold and write row to CSV if it does
    if ratio_sum > sum_threshold:
        with open(output_csv_path, 'a') as f:
            row.to_frame().T.to_csv(f, header=False, index=False)

    print(f"Processed video {idx + 1}/{len(video_data)}: {video_path}")

print("Processing complete. Selected videos saved to:", output_csv_path)