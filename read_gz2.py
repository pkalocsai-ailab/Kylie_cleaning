import gzip
import pickle
import numpy as np

# Function to read a .gz file containing a pickled dictionary
def read_gz_pickle(file_path):
  with gzip.open(file_path, 'rb') as f:
      data = pickle.load(f)
  return data

# Function to print the shape of the data
def print_data_shapes(data_dict):
    for key in ['left_hand']:   # , 'right_hand'
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

# Example usage
file_path = 'tracker_119_segment_000.gz'  # Replace with your actual file path
data_dict = read_gz_pickle(file_path)

# Accessing the dictionary
#print(data_dict['pose'])
#print(data_dict['face'])
#print(data_dict['left_hand'])
#print(data_dict['right_hand'])

# Print the shape of the left and right hand data
print_data_shapes(data_dict)