import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import time
from math import sqrt
import win32api
import pyautogui
import math

# solution APIs
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
 
# opening camera for capturing video 
video = cv2.VideoCapture(0)

# Mediapipe Hand Landmark Model
# calling the hands class from mediapipe.solutions
with mp_hands.Hands(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5) as hands:
     
    while video.isOpened():
        success, frame = video.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        imageHeight, imageWidth, sucsess = image.shape
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
        
        # multi_hand_landmarks method for Finding postion of Hand landmarks  
        lmList = []
        if results.multi_hand_landmarks:    
            myHand = results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy]) 
                        

        if results.multi_hand_landmarks != None:
          for handLandmarks in results.multi_hand_landmarks:
            for point in mp_hands.HandLandmark:
 
                # Normalising Landmarks
                normalizedLandmark = handLandmarks.landmark[point]
                pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
    
                point=str(point)
                
                # Marking Index finger
                if point=='HandLandmark.INDEX_FINGER_TIP':
                 try:
                    indexfingertip_x=pixelCoordinatesLandmark[0]
                    indexfingertip_y=pixelCoordinatesLandmark[1]
                    win32api.SetCursorPos((indexfingertip_x*4,indexfingertip_y*5))
 
                 except:
                    pass
 

            cv2.imshow('Hand Tracking', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
 
video.release()
cv2.destroyAllWindows()
