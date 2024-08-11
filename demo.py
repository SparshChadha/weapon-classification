import os
import shutil
import pandas as pd

# Load the CSV file
train_csv = pd.read_csv("../research paper/valid_file.csv")

# Get unique classes
classes = train_csv['class'].unique()

# Define the base directory where images are stored
base_dir = "../research paper/weapons/valid"

# Define the directory where the organized images will be saved
output_dir = "../research paper/data"

# Create a directory for each class if it doesn't already exist
for cls in classes:
    class_dir = os.path.join(output_dir, cls)
    os.makedirs(class_dir, exist_ok=True)

# Move or copy images to the corresponding class folder
for _, row in train_csv.iterrows():
    img_filename = row['filename']  # Assuming the CSV has a 'filename' column with image names
    img_class = row['class']
    
    src_path = os.path.join(base_dir, img_filename)
    dest_path = os.path.join(output_dir, img_class, img_filename)
    
    if not os.path.exists(dest_path):
        shutil.copy(src_path, dest_path)  # or shutil.move(src_path, dest_path) to move instead of copying

print("Images have been organized into folders.")
