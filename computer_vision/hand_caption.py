import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import requests
from IOT_hand_keypoints.computer_vision.draw_landmarks import draw_landmarks_on_image
from IOT_hand_keypoints.computer_vision.calculate_distance import calculate_distance

cap = cv.VideoCapture(0)
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

isItOn = False
isItStillTouching = False

while True:
    ret, frame = cap.read()

    thumb_landmark_coordinates = [0,0,0]
    index_landmark_coordinates = [0,0,0]


    height, width, channel = frame.shape

    if not ret:
        print(f'Cant recieve frame')
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image =  mp.Image(
        image_format = mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    detection_result = detector.detect(mp_image)

    annotated_image = draw_landmarks_on_image(
        rgb_frame, detection_result
    )
    
    #Showing the coordinates

    distance, thumb_points, index_poits = calculate_distance(detection_result=detection_result,
                                  width=width,
                                  height=height)

    if(distance < 30 and thumb_points[0] > 10): # This corrects the fact that OpenCV always start the next frame with the hand position in (0,0)
        if(isItStillTouching == False):
            if(isItOn == False):
                isItOn = True
                x = requests.get("http://192.168.15.54/27/on")
                print(f'Entrou')
            else:
                isItOn = False
                x = requests.get("http://192.168.15.54/27/off")
        isItStillTouching = True
    else:
        isItStillTouching = False

    print(f'Is it still touching? {isItStillTouching}\nIs it on?: {isItOn}')


    cv.putText(annotated_image, 
               f'Distance: {distance}', # Text
               (100,100), # Coordinates
               cv.FONT_HERSHEY_SIMPLEX, # Font family
               0.5,# Font scale
               (255,0,255), # Color
               cv.LINE_AA)  # Line type

    

    exibition_frame = cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR)
    #cv.putText(exibition_frame, )

    cv.imshow("Hand langmarks", exibition_frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
detector.close()
cv.destroyAllWindows()