import csv
import gzip
import pickle
import os

# Function to read a .gz file containing a pickled dictionary
def read_gz_pickle(file_path):
  with gzip.open(file_path, 'rb') as f:
      data = pickle.load(f)
  return data

# Define thresholds
movement_threshold = 0.03
sign_threshold = 0.2

# Open the original CSV file
with open('kylie.csv', mode='r') as infile:
  reader = csv.DictReader(infile)
  # Prepare to write to the new CSV file
  with open('kylie_dataset_v3.csv', mode='w', newline='') as outfile:
      writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
      writer.writeheader()

      # Process each row in the original CSV
      for row in reader:
          # Modify the video path
          original_path = row['video_path']
          new_path = original_path.replace(
              'G:/Sorenson/mediapipe/', #/home/ec2-user/DATA/kylie_dataset',
              'G:/Sorenson/mediapipe/' #'/mnt/nvme_storeage/DATA/kylie_dataset_keypoints'
          ).replace('.mp4', '.gz')

          # Calculate total frames
          start_frame = int(row['start_frame'])
          end_frame = int(row['end_frame'])
          total_frames = end_frame - start_frame

          # Read the .gz file
          if os.path.exists(new_path):
              data = read_gz_pickle(new_path)

              # Initialize counters
              left_hand_counter = 0
              right_hand_counter = 0

              # Process frames
              for frame in range(start_frame, end_frame):
                  if frame + 1 < len(data['left_hand']) and frame + 1 < len(data['right_hand']):
                      # Compare left hand data points
                      left_hand_current = data['left_hand'][frame]
                      left_hand_next = data['left_hand'][frame + 1]
                      if len(left_hand_current) > 26 and len(left_hand_next) > 26:
                          if (abs(left_hand_current[25] - left_hand_next[25]) > movement_threshold or
                              abs(left_hand_current[26] - left_hand_next[26]) > movement_threshold):
                              left_hand_counter += 1

                      # Compare right hand data points
                      right_hand_current = data['right_hand'][frame]
                      right_hand_next = data['right_hand'][frame + 1]
                      if len(right_hand_current) > 26 and len(right_hand_next) > 26:
                          if (abs(right_hand_current[25] - right_hand_next[25]) > movement_threshold or
                              abs(right_hand_current[26] - right_hand_next[26]) > movement_threshold):
                              right_hand_counter += 1

              # Calculate ratios
              left_hand_ratio = left_hand_counter / total_frames
              right_hand_ratio = right_hand_counter / total_frames
              total_ratio = left_hand_ratio + right_hand_ratio

              # Check if the total ratio exceeds the sign threshold
              if total_ratio > sign_threshold:
                  writer.writerow(row)