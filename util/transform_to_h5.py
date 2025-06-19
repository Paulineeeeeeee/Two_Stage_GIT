import os
import h5py
import jsonlines
import numpy as np
from PIL import Image
from tqdm import tqdm


# Function to get images as tensors
def get_images_tensor(id, data_path, length):
    path = data_path + id
  
    images = []
    for i in range(length):
        image_path = f"{path}-{i}-{length}.jpg"
        if os.path.exists(image_path):
            image = np.array(Image.open(image_path).convert('RGB'))
            images.append(image)
        else:
            return None  # Return None if any image path does not exist

    images_stacked = np.stack(images, axis=0)  # Fix `dim` to `axis` since numpy uses axis
    return images_stacked

# Create the HDF5 file and store data
def create_h5_dataset(jsonl_file, data_path, h5_file_path):
    print("Creating HDF5 dataset...")
    with jsonlines.open(jsonl_file) as reader:
        print("total entries:", len(list(reader)))
    with jsonlines.open(jsonl_file) as reader:
        with h5py.File(h5_file_path, 'a') as h5_file:  # Use 'a' to append to the HDF5 file
            # Loop through each entry in the jsonl file
            for idx, item in enumerate(tqdm(reader, desc="Processing entries")):
                id = item['id']
                length = item['episode_length']
                images_stacked = get_images_tensor(id, data_path, length)
                if images_stacked is None:
                    continue

                # Determine the length category for grouping
                if length <= 8:
                    length_group = 'length_<=8'
                elif length <= 16:
                    length_group = 'length_<=16'
                else:
                    length_group = 'length_>16'

                # Create or access the appropriate group in the HDF5 file
                cat = item['category']
                group_path = f"{cat}/{length_group}/{id}"
                if group_path not in h5_file:
                    group = h5_file.create_group(group_path)

                    group.attrs['category'] = cat
                    group.attrs['id'] = id

                    group.attrs['goal_info'] = item['goal_info']
                    group.attrs['episode_length'] = length

                    group.create_dataset('images', data=images_stacked, compression='gzip')
                else:
                    print(f"Group path already exists: {group_path}")

# Example usage
category = ['google_apps','general','install','web_shopping']
for cat in category:
    jsonl_file = f'/home/pauline/GIT/preprocessing/no_miss_{cat}_train.jsonl'
    data_path = f'/data/pauline/no-miss-AITW/{cat}/'
    h5_file_path = '/data/pauline/AITW.h5'
    create_h5_dataset(jsonl_file, data_path, h5_file_path)
