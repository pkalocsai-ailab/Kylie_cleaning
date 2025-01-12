import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands and Drawing modules
mp_hands = mp.solutions.hands

# Directory containing video files
video_directory = 'G:/Sorenson/kylie_examples/Bad/Bad_to_Good'
#video_directory = 'G:/Sorenson/kylie_examples/Confirmed_Bad/SmallBad'
#video_directory = 'G:/Sorenson/kylie_examples/Good/small'

# List to store the hand ratios for each video
hand_ratios = []

# Threshold for detecting significant movement
threshold = 0.03  # Adjust this value as needed

# Process each video in the directory
for idx, video_file in enumerate(os.listdir(video_directory)):
    if not video_file.endswith('.mp4'):
        continue

    video_path = os.path.join(video_directory, video_file)
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
                        if prev_left_x is None or abs(x - prev_left_x) > threshold or abs(y - prev_left_y) > threshold:
                            left_hand_coords.append([x, y])
                        prev_left_x, prev_left_y = x, y

                    elif label == 'Right':
                        if prev_right_x is None or abs(x - prev_right_x) > threshold or abs(y - prev_right_y) > threshold:
                            right_hand_coords.append([x, y])
                        prev_right_x, prev_right_y = x, y

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()

    left_rows = len(left_hand_coords)
    right_rows = len(right_hand_coords)

    left_hand_ratio = left_rows / total_frames if total_frames > 0 else 0
    right_hand_ratio = right_rows / total_frames if total_frames > 0 else 0

    print(f"Processed video {idx + 1}: {video_file}")
