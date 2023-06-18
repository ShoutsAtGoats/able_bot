import cv2
import mediapipe as mp
import numpy as np
import rtmidi
from rtmidi.midiconstants import (CONTROL_CHANGE)
import time

midiout = rtmidi.MidiOut()
midiout.open_port(2)

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Tempo in seconds
TEMPO = 0.2

def convert_range(value, in_min, in_max, out_min, out_max):
    l_span = in_max - in_min
    r_span = out_max - out_min
    scaled_value = (value - in_min) / l_span
    scaled_value = out_min + (scaled_value * r_span)
    return np.round(scaled_value)


def send_notes(pitch=60, repeat=1):
    for i in range(repeat):
        note_on = [0x90, pitch, 112]
        note_off = [0x80, pitch, 0]
        midiout.send_message(note_on)
        time.sleep(TEMPO)
        midiout.send_message(note_off)

def send_mod(cc=1, value=0):
    mod1 = ([CONTROL_CHANGE | 0, cc, value])
    print(value)
    if value > 0.0:
        midiout.send_message(mod1)


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
                send_mod(1, v1)
            elif pink_x * w > 540:
                print('Right, MIDI Notes')
                v2 = convert_range(pink_y, 1.0, -1.0, 60, 92)
                send_notes(v2, 1)
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)
    fps = 1
    cv2.putText(img, str(fps), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 200, 5), 3)
    cv2.imshow("Your Face goes here", img)
    cv2.waitKey(fps)
