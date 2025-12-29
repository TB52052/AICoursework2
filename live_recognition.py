#first training the model using the most accurate model from all tasks 
import pandas as pd
import sklearn.neighbors as nb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

#train model 
knn_model = nb.KNeighborsClassifier(
    n_neighbors=6,
    metric='euclidean'
)
knn_model.fit(X_train_scaled, y_train)

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
            prediction = knn_model.predict(feature_vector_scaled)[0]

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
