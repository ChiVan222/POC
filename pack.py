import h5py
import torch
import json
import os
import numpy as np
from tqdm import tqdm

VG150_PREDICATES = [
    "background", "above", "across", "against", "along", "and", "at", 
    "attached to", "behind", "belonging to", "between", "carrying", 
    "covered in", "covering", "eating", "flying in", "for", "from", 
    "growing on", "hanging from", "has", "holding", "in", "in front of", 
    "laying on", "looking at", "lying on", "made of", "mounted on", "near", 
    "of", "on", "on back of", "over", "painted on", "parked on", "part of", 
    "playing", "riding", "says", "sitting on", "standing on", "to", "under", 
    "using", "walking in", "walking on", "watching", "wearing", "wears", "with"
]

SPATIAL_CLASSES = [
    "above", "across", "against", "along", "at", "behind", "between",
    "in", "in front of", "near", "on", "on back of", "over", "under", "with"
]

ACTION_CLASSES = [
    "carrying", "eating", "flying in", "holding", "laying on",
    "looking at", "lying on", "playing", "riding", "sitting on",
    "standing on", "using", "walking in", "walking on",
    "watching", "wearing", "wears"
]

SPATIAL_MAP = {n: i for i, n in enumerate(SPATIAL_CLASSES)}
ACTION_MAP = {n: i for i, n in enumerate(ACTION_CLASSES)}

GLOBAL_TO_SPATIAL = {i: SPATIAL_MAP.get(n, -100) for i, n in enumerate(VG150_PREDICATES)}
GLOBAL_TO_ACTION  = {i: ACTION_MAP.get(n, -100)  for i, n in enumerate(VG150_PREDICATES)}


def merge_using_lookup(split="train"):
    print(f"{'='*60}")
    print(f" MERGING SPLIT: {split}")
    print(f"{'='*60}")
    SRC_DIR = f"vg_data/{split}_features_final"
    DST_FILE = f"vg_data/{split}_features_negatives.h5"
    LOOKUP_FILE = "vg_data/lookup_map.json"
    if os.path.exists(DST_FILE):
        print(f" Output file already exists: {DST_FILE}")
        return

    print(f">>> Loading lookup map: {LOOKUP_FILE}")
    with open(LOOKUP_FILE, "r") as f:
        full_map = json.load(f)

    if split not in full_map:
        raise ValueError(f"Split '{split}' not found in lookup map")

    split_map = full_map[split]

    indices = sorted(int(k) for k in split_map.keys())
    max_index = indices[-1] if indices else 0
    num_images = max_index + 1

    print(f">>> Found {len(indices)} entries, max index = {max_index}")

    success, missing = 0, 0

    with h5py.File(DST_FILE, "w") as h5:
        h5.attrs["split"] = split
        h5.attrs["num_images"] = num_images

        for dataset_idx in tqdm(range(num_images), desc=f"Packing {split}"):
            str_idx = str(dataset_idx)

            if str_idx not in split_map:
                missing += 1
                continue

            file_idx = split_map[str_idx]
            pt_path = os.path.join(SRC_DIR, f"{file_idx}.pt")

            if not os.path.exists(pt_path):
                missing += 1
                continue

            try:
                triplets = torch.load(pt_path, map_location="cpu")
                if not triplets:
                    missing += 1
                    continue

                geo, vis, s_labels, a_labels, preds = [], [], [], [], []

                for t in triplets:
                    pid = int(t["pred"])
                    if pid < 0 or pid >= len(VG150_PREDICATES):
                        continue

                    geo.append(t["geo"])
                    vis.append(t["vis"])
                    s_labels.append(GLOBAL_TO_SPATIAL[pid])
                    a_labels.append(GLOBAL_TO_ACTION[pid])
                    preds.append(pid)

                if not geo:
                    missing += 1
                    continue

                geo = torch.stack(geo).numpy().astype(np.float32)
                vis = torch.cat(vis, dim=0).numpy().astype(np.float16)

                s_labels = np.asarray(s_labels, dtype=np.int16)
                a_labels = np.asarray(a_labels, dtype=np.int16)
                preds    = np.asarray(preds,    dtype=np.int16)

                grp = h5.create_group(str_idx)
                grp.create_dataset("geo", data=geo, compression="lzf")
                grp.create_dataset("vis", data=vis, compression="lzf")
                grp.create_dataset("s_label", data=s_labels)
                grp.create_dataset("a_label", data=a_labels)
                grp.create_dataset("pred", data=preds)
                grp.attrs["num_rel"] = geo.shape[0]

                success += 1

            except Exception as e:
                print(f"⚠️ Error on {pt_path}: {e}")
                missing += 1

    print("\n Merge Complete")
    print(f"   Split          : {split}")
    print(f"   Packed Images  : {success}")
    print(f"   Missing/Empty  : {missing}")
    print(f"   Output Size    : {os.path.getsize(DST_FILE) / (1024**3):.2f} GB")


if __name__ == "__main__":
    merge_using_lookup(split="train")
    merge_using_lookup(split="test")
