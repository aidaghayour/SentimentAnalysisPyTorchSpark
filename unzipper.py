import zipfile
import os

def unzip_file(zip_path, extract_to_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_folder)

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the paths
zip_path = os.path.join(current_directory, 'sentiment140.zip')  # Assuming the zip file is named 'dataset.zip'
extract_to_folder = os.path.join(current_directory, 'sentiment140.csv')

# Ensure the destination directory exists
os.makedirs(extract_to_folder, exist_ok=True)

# Extract the file
unzip_file(zip_path, extract_to_folder)
print(f"Extracted files to {extract_to_folder}")
