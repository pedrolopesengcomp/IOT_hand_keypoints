import numpy as np

def calculate_distance(detection_result, width, height):
    thumb_landmark_coordinates = [0,0,0]
    index_landmark_coordinates = [0,0,0]

    distance = 0
    thumb_points = [0,0]
    index_points = [0,0]

    for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):


            thumb_landmark = hand_landmarks[4]
            index_landmark = hand_landmarks[8]

            # Normalized coordinates
            thumb_x = thumb_landmark.x
            thumb_y = thumb_landmark.y
            thumb_z = thumb_landmark.z


            thumb_x_absolute = int(thumb_x * width)
            thumb_y_absolute = int(thumb_y * height)

            thumb_points = [thumb_x_absolute, thumb_y_absolute]

            thumb_landmark_coordinates[0] = thumb_x_absolute
            thumb_landmark_coordinates[1] = thumb_y_absolute
            thumb_landmark_coordinates[2] = thumb_z

            index_x = index_landmark.x
            index_y = index_landmark.y
            index_z = index_landmark.z


            index_x_absolute = int(index_x * width)
            index_y_absolute = int(index_y * height)

            index_points = [index_x_absolute, index_y_absolute]

            index_landmark_coordinates[0] = index_x_absolute
            index_landmark_coordinates[1] = index_y_absolute
            index_landmark_coordinates[2] = thumb_z

            distance_x = int(abs(index_x_absolute - thumb_x_absolute))
            distance_y = int(abs(index_y_absolute - thumb_y_absolute))

            distance_array = np.array([distance_x, distance_y])

            distance = np.linalg.norm(distance_array) # Distance between the index finger and the thumb

    return distance, thumb_points, index_points
