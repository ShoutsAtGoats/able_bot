import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

def convert_range(value, in_min, in_max, out_min, out_max):
    l_span = in_max - in_min
    r_span = out_max - out_min
    scaled_value = (value - in_min) / l_span
    scaled_value = out_min + (scaled_value * r_span)
    return np.round(scaled_value)


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        h, w, c = img.shape
        for hand_landmarks in results.multi_hand_landmarks:
            pink_x = hand_landmarks.landmark[mpHands.HandLandmark.PINKY_TIP].x
            pink_y = hand_landmarks.landmark[mpHands.HandLandmark.PINKY_TIP].y
            if pink_x * w < 540:
                print("Left, MIDI CC DATA")
                v1 = convert_range(pink_y, 1.0, 0.0, 0, 127)
                print(v1)
            elif pink_x * w > 540:
                print('Right, MIDI Notes')
                v2 = convert_range(pink_y, 1.0, -1.0, 60, 92)
                print(v2)
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)


    fps = 1
    cv2.putText(img, str(fps), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 200, 5), 3)
    cv2.imshow("Your Face goes here", img)
    cv2.waitKey(fps)
