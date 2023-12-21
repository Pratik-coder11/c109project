import cv2
import mediapipe as mp
import numpy as np
import pyautogui

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

tipIds = [4, 8, 12, 16, 20]

def are_all_fingers_folded(hand_landmarks):
    if hand_landmarks:
        landmarks = hand_landmarks[0].landmark
        fingers_folded = all(landmarks[i].y > landmarks[i - 2].y for i in tipIds[1:])
        return fingers_folded
    return False

while True:
    success, image = cap.read()
    image = cv2.flip(image, 1)

    results = hands.process(image)

    hand_landmarks = results.multi_hand_landmarks

    if hand_landmarks:
        mp_drawing.draw_landmarks(image, hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    if are_all_fingers_folded(hand_landmarks):
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)
        screenshot_cv2 = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite("screenshot.png", screenshot_cv2)
        print("Screenshot captured!")

    cv2.imshow("Media Controller", image)

    key = cv2.waitKey(1)
    if key == 32:
        break

cv2.destroyAllWindows()
