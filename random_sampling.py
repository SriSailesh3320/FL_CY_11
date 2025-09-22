# prepare_sampled_dataset.py
import os
import random
import shutil

# --- Configuration ---
# The source of your high-quality, preprocessed images
SOURCE_DIR = 'data/Brain_MRI/cleaned/Training/'
# The new folder where the 100-images-per-class dataset will be created
DEST_DIR = 'data/Brain_MRI/sampled_dataset/Training'
SAMPLES_PER_CLASS = 200

print(f"Starting dataset sampling...")
print(f"Source: {SOURCE_DIR}")
print(f"Destination: {DEST_DIR}")

# Create the main destination directory
if os.path.exists(DEST_DIR):
    shutil.rmtree(DEST_DIR) # Remove old directory to ensure a fresh start
os.makedirs(DEST_DIR)

# Get the list of class subdirectories (e.g., 'glioma', 'meningioma')
class_dirs = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

# Loop through each class directory
for class_name in class_dirs:
    print(f"\nProcessing class: {class_name}")
    
    source_class_path = os.path.join(SOURCE_DIR, class_name)
    dest_class_path = os.path.join(DEST_DIR, class_name)
    os.makedirs(dest_class_path) # Create class subfolder in the destination
    
    # Get a list of all image files in the source class folder
    all_images = [f for f in os.listdir(source_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Check if there are enough images to sample from
    if len(all_images) < SAMPLES_PER_CLASS:
        print(f"  WARNING: Only {len(all_images)} images found. Using all of them.")
        sampled_images = all_images
    else:
        # Randomly sample 100 images
        random.seed(42) # Use a seed for reproducible results
        sampled_images = random.sample(all_images, SAMPLES_PER_CLASS)
        print(f"  Randomly selected {len(sampled_images)} images.")

    # Copy the sampled images to the new directory
    for image_name in sampled_images:
        source_image_path = os.path.join(source_class_path, image_name)
        dest_image_path = os.path.join(dest_class_path, image_name)
        shutil.copy(source_image_path, dest_image_path)
        
print("\nDataset sampling complete!")
print(f"New dataset is ready at: {DEST_DIR}")