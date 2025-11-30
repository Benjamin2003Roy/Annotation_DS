import os
import pickle
import uuid
import json
import cv2

input_dir = "/Users/benjaminroyv/Downloads/data_something/data/c57654ee-8d89-4d4a-aa8f-5d173a69abea/ef5af20d-e922-4f1a-add1-343b71731368/wbc/Recon_data/focused_fov_pkl"

for pkl_file in os.listdir(input_dir):
    if pkl_file.endswith(".pkl"):
        pkl_path = os.path.join(input_dir, pkl_file)
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        image = cv2.imwrite("/Users/benjaminroyv/Downloads/data_something/data/c57654ee-8d89-4d4a-aa8f-5d173a69abea/ef5af20d-e922-4f1a-add1-343b71731368/wbc/Recon_data/focused_fov_pkl_1/" + pkl_file.replace(".pkl", ".png"), data)