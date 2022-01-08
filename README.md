# Gesture-Cursor-Control

## 💾 REQUIREMENTS
+ opencv-python
+ mediapipe
+ numpy
+ win32api
+ pyautogui

```bash
pip install -r requirements.txt
```
***
### MEDIAPIPE
<div align="center">
  <img alt="mediapipeLogo" src="images/mediapipe.png" />
</div>

> MediaPipe offers open source cross-platform, customizable ML solutions for live and streaming media.

#### Hand Landmark Model
After the palm detection over the whole image our subsequent hand landmark model performs precise keypoint localization of 21 3D hand-knuckle coordinates inside the detected hand regions via regression, that is direct coordinate prediction. The model learns a consistent internal hand pose representation and is robust even to partially visible hands and self-occlusions.

To obtain ground truth data, we have manually annotated ~30K real-world images with 21 3D coordinates, as shown below (we take Z-value from image depth map, if it exists per corresponding coordinate). To better cover the possible hand poses and provide additional supervision on the nature of hand geometry, we also render a high-quality synthetic hand model over various backgrounds and map it to the corresponding 3D coordinates.<br>

#### Solution APIs
##### Configuration Options
> Naming style and availability may differ slightly across platforms/languages.

+ <b>STATIC_IMAGE_MODE</b><br>
If set to false, the solution treats the input images as a video stream. It will try to detect hands in the first input images, and upon a successful detection further localizes the hand landmarks. In subsequent images, once all max_num_hands hands are detected and the corresponding hand landmarks are localized, it simply tracks those landmarks without invoking another detection until it loses track of any of the hands. This reduces latency and is ideal for processing video frames. If set to true, hand detection runs on every input image, ideal for processing a batch of static, possibly unrelated, images. Default to false.

+ <b>MAX_NUM_HANDS</b><br>
Maximum number of hands to detect. Default to 2.

+ <b>MODEL_COMPLEXITY</b><br>
Complexity of the hand landmark model: 0 or 1. Landmark accuracy as well as inference latency generally go up with the model complexity. Default to 1.

+ <b>MIN_DETECTION_CONFIDENCE</b><br>
Minimum confidence value ([0.0, 1.0]) from the hand detection model for the detection to be considered successful. Default to 0.5.

+ <b>MIN_TRACKING_CONFIDENCE:</b><br>
Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model for the hand landmarks to be considered tracked successfully, or otherwise hand detection will be invoked automatically on the next input image. Setting it to a higher value can increase robustness of the solution, at the expense of a higher latency. Ignored if static_image_mode is true, where hand detection simply runs on every image. Default to 0.5.

<br>

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
import time
from math import sqrt
import win32api
import pyautogui
import math
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
Normalising Landmarks
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
