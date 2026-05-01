import zipfile
import os

zip_path = r"d:/Ancient-Tamil-Script-Recognition-master/Ancient-Tamil-Script-Recognition-master/Labels/Labelled Dataset - Fig 51.zip"
extract_path = r"d:/Ancient-Tamil-Script-Recognition-master/Ancient-Tamil-Script-Recognition-master/Labels/Labelled Dataset - Fig 51"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction complete.")
