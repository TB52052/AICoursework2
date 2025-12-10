import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sklearn.model_selection import train_test_split

import os
import csv

import pandas as pd
import numpy as np

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
header = ["gesture", "filename", "hand"]
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
                    
                    hand = result.handedness[idx][0].category_name #LEFT or RIGHT hand
                    
                    row = [folder, filename, hand]

                    # Flatten landmarks x,y,z
                    for lm in hand_landmarks:
                        #this ensures correct number formatting
                        row += [f'{lm.x:.8f}', f'{lm.y:.8f}', f'{lm.z:.8f}']
                    
                    
                    #only submits hands that are the RIGHT hands
                    if hand == "Right":
                        #write hand co-ordinate to .csv file
                        writer.writerow(row)
                    
print("CSV file saved to:", output_csv)
'''
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
'''

df = pd.read_csv('hand_landmarks.csv')
#print(data)

data = df.drop(['filename', 'hand'],axis=1)

#group by gesture
grouped = data.groupby("gesture")

#compute the mean of each column in the .csv (returns 63 means and sd)
gesture_mean = grouped.mean(numeric_only=True)
gesture_std = grouped.std(numeric_only=True, ddof=1)

#keep data within 3 standard deviations
k = 3 
#Standard Deviation was orginally set to 2, however that means only 2344/3648 would be kept
#the decision has been made to keep k = 3 to ensure sufficient data for training 

valid_rows = []

for idx, row in data.iterrows():
    gesture = row["gesture"]

    #drop gesture column to have only numeric values
    values = row.drop("gesture").to_numpy(dtype=float)

    #calculating the mean and std for each gesture
    mean_vals = gesture_mean.loc[gesture].to_numpy(dtype=float)
    std_vals  = gesture_std.loc[gesture].to_numpy(dtype=float)

    # comparing against all values for that gesture (within 3 std)
    diff = np.abs(values - mean_vals)
    within_range = diff <= (k * std_vals)

    #if within range then added to valid rows list
    if np.all(within_range):
        valid_rows.append(row)

#creates a new .csv with the "valid" rows
valid_df = pd.DataFrame(valid_rows)
valid_df.to_csv("hand_landmarks_valid.csv", index=False)

print("Original:", len(data))
print("Kept:", len(valid_df))



