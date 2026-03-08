import os
import pickle

data_path = os.path.join("things_data", "osfstorage", "images_THINGS", "object_images")

folders = [
    f for f in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, f))
]

all_images = {}

for folder in folders:
    folder_path = os.path.join(data_path, folder)
    paths = [
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if filename.lower().endswith((".jpg",))
    ]
    all_images[folder] = paths

with open("all_images.pkl", "wb") as f:
    pickle.dump(all_images, f)
