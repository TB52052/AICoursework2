#first training the model using the most accurate model from all tasks 
import pandas as pd
import sklearn.neighbors as nb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import math
import numpy as np
from collections import Counter

#load dataset
data = pd.read_csv('hand_landmarks_valid.csv')

#splitting into different sets for training, testing and validating
#weighting of each is train approx 70%, test 20% and val 10% of data set
train_val, test = train_test_split(data, test_size=0.2, random_state=42)
train, val = train_test_split(train_val, test_size=0.125, random_state=42)

#setting training label and data
X_train = train.drop(['gesture'],axis=1).to_numpy()
y_train = train['gesture'].to_numpy()


#implementating a scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# distance functions
def get_euclidean(v1, v2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

def get_manhattan(v1, v2):
    return sum(abs(a - b) for a, b in zip(v1, v2))

# Manual KNN
def knn(X_train, y_train, X_test, k=3, distance_metric='euclidean'):
    
    if distance_metric == 'euclidean':
        dist_func = get_euclidean
    elif distance_metric == 'manhattan':
        dist_func = get_manhattan
        
    predictions = []

    for x1 in X_test:
        # computing the distance to all training points (where x1 is test and x2 is train)
        distances = [(dist_func(x1, x2), label) 
                     for x2, label in zip(X_train, y_train)]
        
        #sort by distance
        distances = sorted(distances)
        
        #selecting the k nearest
        k_nearest = distances[:k]
        
        #inspecting the most common label in the k nearest
        k_labels = [y for _, y in k_nearest] #just taking the label
        #1 is top of list and [0][0] leaves just the label 
        most_common = Counter(k_labels).most_common(1)[0][0] 
        predictions.append(most_common)
    
    return predictions




#taken from original mediapipe implementation and used to convert points into data coordinates
def landmarks_to_feature_vector(hand_landmarks):
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return features

#implementation of mediapipe and video functionality
import cv2
import mediapipe as mp
import numpy as np

#mediapipe live video implementation
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #draw landmarks on hand
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            #extraction of landmarks and format to match data extraction
            feature_vector = landmarks_to_feature_vector(hand_landmarks)
            feature_vector = np.array(feature_vector).reshape(1, -1)

            #using the above scaler and then forming a prediction with the trained model
            feature_vector_scaled = scaler.transform(feature_vector)
            prediction = knn(
    X_train_scaled,
    y_train,
    feature_vector_scaled,
    k=6,
    distance_metric='euclidean'
)[0]

            #displaying prediction as label
            cv2.putText(
                frame,
                f"Gesture: {prediction}",
                (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

    cv2.imshow("Live ASL Gesture Recognition", frame)

    #to close the video recognition
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

cap.release()
cv2.destroyAllWindows()
