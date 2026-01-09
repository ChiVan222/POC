import h5py
import numpy as np

def inspect_h5_structure(file_path, num_samples=1):
    """
    Opens an H5 file and prints its attributes and internal dataset structure.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"{'='*60}")
            print(f"FILE: {file_path}")
            print(f"{'='*60}")

            print("\n>>> ROOT ATTRIBUTES:")
            if f.attrs.keys():
                for attr_name, attr_value in f.attrs.items():
                    print(f"    - {attr_name}: {attr_value}")
            else:
                print("    (No root attributes found)")

            keys = list(f.keys())
            total_keys = len(keys)
            print(f"\n>>> TOTAL GROUPS/KEYS: {total_keys}")

            if total_keys == 0:
                print("    (The file appears to be empty)")
                return

            print(f"\n>>> INSPECTING {min(num_samples, total_keys)} SAMPLE(S):")
            for i in range(min(num_samples, total_keys)):
                key = keys[i]
                item = f[key]
                
                print(f"\n    [Index: '{key}']")
                
                if isinstance(item, h5py.Group):
                    if item.attrs.keys():
                        print("      Attributes:")
                        for attr_name, attr_value in item.attrs.items():
                            print(f"        - {attr_name}: {attr_value}")
                    
                    print("      Datasets:")
                    for dset_name in item.keys():
                        dset = item[dset_name]
                        print(f"        - {dset_name:10} | Shape: {str(dset.shape):15} | Dtype: {dset.dtype}")
                
                elif isinstance(item, h5py.Dataset):
                    print(f"      Type: Direct Dataset | Shape: {item.shape} | Dtype: {item.dtype}")

            print(f"\n{'='*60}")
            
    except Exception as e:
        print(f"Error reading H5 file: {e}")

if __name__ == "__main__":
    FILE_PATH = "vg_data/test_features.h5"
    inspect_h5_structure(FILE_PATH, num_samples=1)