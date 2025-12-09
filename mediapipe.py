import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sklearn.model_selection import train_test_split

import os
import csv

import pandas as pd

model_path = r"C:\Users\taylo\Documents\UNI\Modules\CMP-6058A-25-SEM1-A - Artificial Intelligence\Coursework 2\model_folder\hand_landmarker.task"
dataset_path = r"C:\Users\taylo\Documents\UNI\Modules\CMP-6058A-25-SEM1-A - Artificial Intelligence\Coursework 2\Datasets\CW2_dataset_final"
output_csv = r"C:\Users\taylo\Documents\UNI\Modules\CMP-6058A-25-SEM1-A - Artificial Intelligence\Coursework 2\hand_landmarks.csv"
output_world_csv = r"C:\Users\taylo\Documents\UNI\Modules\CMP-6058A-25-SEM1-A - Artificial Intelligence\Coursework 2\world_hand_landmarks.csv"


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

#capturing the label data
header = ["gesture", "filename",]
#capturing the positional coordinates
for i in range(21):
    header += [f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"]

#capturing Landmarks 

#open csv for writing to 
with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    #create handlandmarker 
    #taken from Hand landmarks detection guide for Python (google - link provided in CW)
    with HandLandmarker.create_from_options(options) as landmarker:

        #loop through all folders a-j
        for folder in sorted(os.listdir(dataset_path)):
            folder_path = os.path.join(dataset_path, folder)
            
            #loop through contents of each folder
            for filename in os.listdir(folder_path):
                #will only process .jpg files
                if not filename.lower().endswith((".jpg")):
                    continue

                file_path = os.path.join(folder_path, filename)
                #processing image
                image = mp.Image.create_from_file(file_path)

                #running the hand detection on the selected image
                result = landmarker.detect(image)

                #saving each piece of information to .csv file 
                for idx, hand_landmarks in enumerate(result.hand_landmarks):
                    row = [folder, filename]

                    # Flatten landmarks x,y,z
                    for lm in hand_landmarks:
                        #this ensures correct number formatting
                        row += [f'{lm.x:.8f}', f'{lm.y:.8f}', f'{lm.z:.8f}']
                    
                    #write hand co-ordinate to .csv file
                    writer.writerow(row)
                    
print("CSV file saved to:", output_csv)

#capturing global landmarks 

#open csv for writing to 
with open(output_world_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    #create handlandmarker 
    #taken from Hand landmarks detection guide for Python (google - link provided in CW)
    with HandLandmarker.create_from_options(options) as landmarker:

        #loop through all folders a-j
        for folder in sorted(os.listdir(dataset_path)):
            folder_path = os.path.join(dataset_path, folder)
            
            #loop through contents of each folder
            for filename in os.listdir(folder_path):
                #will only process .jpg files
                if not filename.lower().endswith((".jpg")):
                    continue

                file_path = os.path.join(folder_path, filename)
                #processing image
                image = mp.Image.create_from_file(file_path)

                #running the hand detection on the selected image
                result = landmarker.detect(image)

                #saving each piece of information to .csv file 
                for idx, hand_world_landmarks in enumerate(result.hand_world_landmarks):
                    row = [folder, filename]

                    # Flatten landmarks x,y,z
                    for lm in hand_world_landmarks:
                        row += [lm.x, lm.y, lm.z]
                    
                    #write hand co-ordinate to .csv file
                    writer.writerow(row)

print("CSV file saved to:", output_world_csv)


data = pd.read_csv('hand_landmarks.csv')
#print(data)
