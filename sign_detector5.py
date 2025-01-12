import csv
import gzip
import pickle
import os

input_csv_file = '/home/ec2-user/sora/VSL-SLT/src/metadata/kylie_dataset_ASL_SLT_v2.csv'
output_csv_file = '/home/ec2-user/peter/kylie_clean/kylie_dataset_ASL_SLT_v3.csv'

# Function to read a .gz file containing a pickled dictionary
def read_gz_pickle(file_path):
  with gzip.open(file_path, 'rb') as f:
      data = pickle.load(f)
  return data

# Define thresholds
movement_threshold = 0.02
sign_threshold = 0.2

# Initialize counters
bad_frame_spec_counter = 0
no_gzip_counter = 0

# Open the original CSV file
with open(input_csv_file, mode='r') as infile:
  reader = csv.DictReader(infile)
  rows = list(reader)  # Convert to list to get the total number of rows
  total_files = len(rows)

  # Prepare to write to the new CSV file
  with open(output_csv_file, mode='w', newline='') as outfile:
      writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
      writer.writeheader()

      # Process each row in the original CSV
      for index, row in enumerate(rows, start=1):

          if index == 11:
            break

          # Modify the video path
          original_path = row['video_path']
          new_base_path = original_path.replace(
              '/home/ec2-user/DATA/kylie_dataset',
              '/mnt/nvme_storeage/DATA/kylie_dataset_keypoints'
          )
          directory, filename = os.path.split(new_base_path)
          new_directory = os.path.join(directory, "mediapipe_kp")
          filename_without_ext = os.path.splitext(filename)[0]
          new_filename = f"{filename_without_ext}.gz"
          new_path = os.path.join(new_directory, new_filename)
        #   print(original_path)
        #   print(new_path)
        #   breakpoint()

          # Calculate total frames
          start_frame = int(row['start_frame'])
          end_frame = int(row['end_frame'])
          total_frames = end_frame - start_frame

          # Check for invalid frame specifications
          if total_frames <= 0:
              bad_frame_spec_counter += 1
              continue

          # Read the .gz file
          if not os.path.exists(new_path):
              no_gzip_counter += 1
              continue

          data = read_gz_pickle(new_path)

          # Initialize counters
          left_hand_counter = 0
          right_hand_counter = 0

          # Process frames
          for frame in range(start_frame, end_frame - 1):
              left_hand_current = data['left_hand'][frame] if frame < len(data['left_hand']) else []
              left_hand_next = data['left_hand'][frame + 1] if frame + 1 < len(data['left_hand']) else []
              right_hand_current = data['right_hand'][frame] if frame < len(data['right_hand']) else []
              right_hand_next = data['right_hand'][frame + 1] if frame + 1 < len(data['right_hand']) else []

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

          # Print progress
          print(f"Processed {index}th file out of {total_files}")

# Print the counters
print(f"Number of rows with bad frame specifications: {bad_frame_spec_counter}")
print(f"Number of missing .gz files: {no_gzip_counter}")