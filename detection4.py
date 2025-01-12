import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the video file
#video_path = 'Good.mp4'
#video_path = 'G:/Sorenson/kylie_examples/Confirmed_Bad/tracker_52_segment_008.mp4'
video_path = 'G:/Sorenson/kylie_examples/Good/tracker_1_segment_022.mp4'
cap = cv2.VideoCapture(video_path)

# Matrices to store x, y coordinates of the first landmark for left and right hands
left_hand_coords = []
right_hand_coords = []

# Count total frames
total_frames = 0

# Set up the hand detector
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Reached the end of the video or failed to load.")
            break

        total_frames += 1

        # Convert the BGR image to RGB before processing.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Process detected hands
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get the label of handedness (left or right)
                label = handedness.classification[0].label
                
                # Get x, y coordinates of the first landmark (index 0)
                x = hand_landmarks.landmark[8].x
                y = hand_landmarks.landmark[8].y
                
                if label == 'Left':
                    left_hand_coords.append([x, y])
                elif label == 'Right':
                    right_hand_coords.append([x, y])
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Convert lists to numpy arrays
left_hand_matrix = np.array(left_hand_coords)
right_hand_matrix = np.array(right_hand_coords)

# Calculate number of rows and standard deviations for left hand matrix
left_rows = left_hand_matrix.shape[0]
left_std_column1 = np.var(left_hand_matrix[:, 0]) if left_rows > 0 else 0
left_std_column2 = np.var(left_hand_matrix[:, 1]) if left_rows > 0 else 0

# Calculate number of rows and standard deviations for right hand matrix
right_rows = right_hand_matrix.shape[0]
right_std_column1 = np.var(right_hand_matrix[:, 0]) if right_rows > 0 else 0
right_std_column2 = np.var(right_hand_matrix[:, 1]) if right_rows > 0 else 0

# Calculate ratios of detections to total frames
left_hand_ratio = left_rows / total_frames if total_frames > 0 else 0
right_hand_ratio = right_rows / total_frames if total_frames > 0 else 0

# Print results
print(f"Left Hand - Ratio: {left_hand_ratio}, Var Col1: {left_std_column1}, Var Col2: {left_std_column2}")
print(f"Right Hand - Ratio: {right_hand_ratio}, Var Col1: {right_std_column1}, Var Col2: {right_std_column2}")