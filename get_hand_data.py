

import cv2
import os
import mediapipe as mp
import pandas as pd
import numpy as np



def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = os.path.join(folder,filename)
        images.append(img)
    return images





mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

df_list = []

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    
    for label_idx, folder in enumerate(['a', 'b', 'c']):
        for img in load_images_from_folder(folder):
            image = cv2.imread(img)
            # image = cv2.pyrDown(image)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:

                for hand_landmarks in results.multi_hand_landmarks:
                    handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                    handLabel = results.multi_handedness[handIndex].classification[0].label

                    handLandmarks = []
                    for landmarks in hand_landmarks.landmark:
                        handLandmarks.append([landmarks.x, landmarks.y])

                    df = pd.DataFrame(handLandmarks)
                    df = df.unstack().to_frame().T
                    df.columns = [f'feature_{i}' for i in range(1, 43)]

                    df['label'] = label_idx
                    df_list.append(df)

            #         mp_drawing.draw_landmarks(
            #             image,
            #             hand_landmarks,
            #             mp_hands.HAND_CONNECTIONS,
            #             mp_drawing_styles.get_default_hand_landmarks_style(),
            #             mp_drawing_styles.get_default_hand_connections_style())

            # cv2.imshow('MediaPipe Hands', image)
            # cv2.waitKey(0)




df = pd.concat(df_list)
df.to_csv('hand_dataset.csv', index=False)
