# Gesture-Cursor-Control

## 💾 REQUIREMENTS
+ opencv-python
+ mediapipe
+ numpy
+ win32api


```bash
pip install -r requirements.txt
```
***
### MEDIAPIPE
<div align="center">
  <img alt="mediapipeLogo" src="images/mediapipe.png" />
</div>


Source: [MediaPipe Hands Solutions](https://google.github.io/mediapipe/solutions/hands#python-solution-api)

<div align="center">
    <img alt="mediapipeLogo" src="images/hand_landmarks.png" height="200 x" />
    <img alt="mediapipeLogo" src="images/hand_crops.png" height="360 x" weight ="640 x" />
    <img alt="output" src="images/hand.gif" height="280 x" weight ="140 x" />
    
</div>


## 📝 CODE EXPLANATION
<b>Importing Libraries</b>
```py
import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from math import sqrt
import win32api
```
***
Solution APIs 
```py
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
```

Opening webCam using OpenCV
```py
video = cv2.VideoCapture(0)

```
***
Mediapipe Hand Landmark Model <br/>
Calling the hands class from mediapipe.solutions 
```py
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
```
***
Using multi_hand_landmarks method for Finding postion of Hand landmarks
```py
lmList = []
if results.multi_hand_landmarks:    
    myHand = results.multi_hand_landmarks[0]
    for id, lm in enumerate(myHand.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy]) 

```
***
Checking if the image has been processed
```py
if results.multi_hand_landmarks != None:
  for handLandmarks in results.multi_hand_landmarks:
    for point in mp_hands.HandLandmark:
```

***
Normalising Landmarks (converting decimal value to pixels) 
```
normalizedLandmark = handLandmarks.landmark[point]
pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
```

***
Converting to string
```py
point=str(point)
```
                
***
Assigning variable for Index finger position
```py
if point=='HandLandmark.INDEX_FINGER_TIP':
 try:
    indexfingertip_x=pixelCoordinatesLandmark[0]
    indexfingertip_y=pixelCoordinatesLandmark[1]
    win32api.SetCursorPos((indexfingertip_x*4,indexfingertip_y*5))

 except:
    pass
```

***
Displaying Output using `cv2.imshow` method
```py
cv2.imshow('Hand Tracking', image)
if cv2.waitKey(10) & 0xFF == ord('q'):
  break
```

***
Closing webCam
```py
video.release()
```

***
Closing window
```py
cv2.destroyAllWindows()
```
