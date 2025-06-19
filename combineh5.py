import os
import h5py
import multiprocessing
from tqdm import tqdm  # Optional for progress indication

def list_h5_files(directory):
    """List all .h5 files in the given directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.h5')]

def read_h5_file(file_path, file_index):
    """Read data from an h5 file, appending a file index to each group to keep them unique."""
    data = {}
    with h5py.File(file_path, 'r') as file:
        for group in file:
            new_group = f"{group}_{file_index}"  # Modify the group name to include the file index
            data[new_group] = {}
            for key in file[group]:
                data[new_group][key] = file[group][key][:]  # Keep the dataset key names standard
    return data

def combine_h5_files(file_paths, output_file):
    """Combine multiple h5 files into one, appending file index to groups to avoid overwriting."""
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Pair each file path with its index and map them to the reader function
        tasks = [(file_path, index) for index, file_path in enumerate(file_paths)]
        results = list(tqdm(pool.starmap(read_h5_file, tasks), total=len(file_paths)))

    # Write the results into one combined .h5 file
    with h5py.File(output_file, 'w') as h5out:
        for result in results:
            for group, datasets in result.items():
                grp = h5out.require_group(group)
                for key, data in datasets.items():
                    grp.create_dataset(key, data=data)

directory_path = "/data2/peter/rico"
output_h5_file = "rico.h5"
file_paths = list_h5_files(directory_path)
combine_h5_files(file_paths, output_h5_file)