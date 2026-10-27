import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks

    annotated_image = np.copy(rgb_image)

    # Draw landmarks and connections for each detected hand
    for hand_landmarks in hand_landmarks_list:
        mp.tasks.vision.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        )

    return annotated_image