import gzip
import json

# Function to read a .gz file containing a JSON dictionary
def read_gz_file(file_path):
  with gzip.open(file_path, 'rt', encoding='utf-8') as f:
      data = json.load(f)
  return data

# Example usage
file_path = 'tracker_119_segment_000.gz'  # Replace with your actual file path
data_dict = read_gz_file(file_path)

# Accessing the dictionary
print(data_dict['pose'])
print(data_dict['face'])
print(data_dict['left_hand'])
print(data_dict['right_hand'])