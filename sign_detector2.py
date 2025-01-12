import csv
import gzip
import pickle
import os
import numpy as np 

# Function to read a .gz file containing a pickled dictionary
def read_gz_pickle(file_path):
  with gzip.open(file_path, 'rb') as f:
      data = pickle.load(f)
  return data

# Function to print the shape of the data
def print_data_shapes(data_dict):
  for key in ['left_hand', 'right_hand']:
      data = data_dict.get(key, None)
      if data is not None:
          print(f"\nAnalyzing {key} data:")
          print(f"Type of data: {type(data)}")
          print(f"Length of data: {len(data)}")
          
          if isinstance(data, list):
              print("Structure of the list:")
              for i, item in enumerate(data):
                  print(f"  Item {i}: Type = {type(item)}, ", end="")
                  if isinstance(item, list):
                      print(f"Length = {len(item)}")
                      if item:
                          print(f"    First element type: {type(item[0])}")
                  else:
                      print(f"Value = {item}")
      else:
          print(f"No data found for {key}")

# Define thresholds
movement_threshold = 0.01
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
              'G:/Sorenson/mediapipe/',  # '/home/ec2-user/DATA/kylie_dataset',
              'G:/Sorenson/mediapipe/'  # '/mnt/nvme_storeage/DATA/kylie_dataset_keypoints'
          ).replace('.mp4', '.gz')

          # Calculate total frames
          start_frame = int(row['start_frame'])
          end_frame = int(row['end_frame'])
          total_frames = end_frame - start_frame

          # Read the .gz file
          if os.path.exists(new_path):
              data = read_gz_pickle(new_path)
              #print_data_shapes(data)  # Debugging: Print the structure of the data

              # Initialize counters
              left_hand_counter = 0
              right_hand_counter = 0

              # Process frames
              # Process frames
              for frame in range(start_frame, end_frame - 1):
                  left_hand_current = data['left_hand'][frame] if frame < len(data['left_hand']) else np.array([])
                  left_hand_next = data['left_hand'][frame + 1] if frame + 1 < len(data['left_hand']) else np.array([])
                  right_hand_current = data['right_hand'][frame] if frame < len(data['right_hand']) else np.array([])
                  right_hand_next = data['right_hand'][frame + 1] if frame + 1 < len(data['right_hand']) else np.array([])
                  print(f"  type: {type(left_hand_current)}")
                  print(f"  value: {left_hand_current[25]}")

                  # Compare left hand data points
                  if isinstance(left_hand_current, list) and isinstance(left_hand_next, list):
                      if len(left_hand_current) > 26 and len(left_hand_next) > 26:
                          if (abs(left_hand_current[25] - left_hand_next[25]) > movement_threshold or
                              abs(left_hand_current[26] - left_hand_next[26]) > movement_threshold):
                              left_hand_counter += 1

                  # Compare right hand data points
                  if isinstance(right_hand_current, list) and isinstance(right_hand_next, list):
                      if len(right_hand_current) > 26 and len(right_hand_next) > 26:
                          if (abs(right_hand_current[25] - right_hand_next[25]) > movement_threshold or
                              abs(right_hand_current[26] - right_hand_next[26]) > movement_threshold):
                              right_hand_counter += 1

              # Calculate ratios
              left_hand_ratio = left_hand_counter / total_frames
              right_hand_ratio = right_hand_counter / total_frames
              total_ratio = left_hand_ratio + right_hand_ratio

              # Print the calculated values
              print(f"Processed file: {new_path}")
              print(f"  Left hand counter: {left_hand_counter}")
              print(f"  Right hand counter: {right_hand_counter}")
              print(f"  Total frames: {total_frames}")
              print(f"  Left hand ratio: {left_hand_ratio:.4f}")
              print(f"  Right hand ratio: {right_hand_ratio:.4f}")
              print(f"  Total ratio: {total_ratio:.4f}")

              # Check if the total ratio exceeds the sign threshold
              if total_ratio > sign_threshold:
                  writer.writerow(row)