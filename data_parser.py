import os
import pickle
from PIL import Image

data_path = os.path.join("things_data", "osfstorage", "images_THINGS", "object_images")

folders = [
    f for f in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, f))
]

all_images = {}

for folder in folders:
    folder_path = os.path.join(data_path, folder)
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                images.append(img.copy())    
    all_images[folder] = images
    print(f"{folder}: {len(images)} images")


with open("all_images.pkl", "wb") as f:
    pickle.dump(all_images, f)
