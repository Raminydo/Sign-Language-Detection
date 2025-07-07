import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


dataset = pd.read_csv('hand_dataset.csv')
X = dataset.drop('label', axis=1)
y = dataset['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn_model = KNeighborsClassifier(n_neighbors=7, weights='distance')
knn_model.fit(X_scaled, y)



mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            handIndex = results.multi_hand_landmarks.index(hand_landmarks)
            handLabel = results.multi_handedness[handIndex].classification[0].label
            
            features = []
            for landmarks in hand_landmarks.landmark:
                features.append([landmarks.x, landmarks.y])

            
            X_test = pd.DataFrame(features)
            X_test = X_test.unstack().to_frame().T
            X_test.columns = [f'feature_{i}' for i in range(1, 43)]
            # X_test = df.values
            X_test_scaled = scaler.transform(X_test)

        
            pred = knn_model.predict(X_test)
            prob = max(knn_model.predict_proba(X_test_scaled)[0])


            if pred == 0 and prob > 0.7:
                gesture_text = 'A'
            elif pred == 1 and prob > 0.7:
                gesture_text = 'B'
            elif pred == 2 and prob > 0.7:
                gesture_text = 'C'
            else:
                gesture_text = ''
  
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, gesture_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Hand Gesture Detection', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()