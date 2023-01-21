import cv2
import mediapipe as mp
import math
import numpy as np

def open_len(arr):
    y_arr = []

    for _,y in arr:
        y_arr.append(y)

    min_y = min(y_arr)
    max_y = max(y_arr)

    return max_y - min_y

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)


RIGHT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)


    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    '''
    print("Nose","x",results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x)
    print("Left eye", results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE_OUTER].x)
    print("Right eye", results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE_OUTER].x)
    print("Right Index", results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX].x)
    print("Nose","y",results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y)
    print("Left Eye", results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE_OUTER].y)
    print("Right eye", results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE_OUTER].y)
    print("Right Index", results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX].y)
    '''
    keypoint_nose_x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x
    keypoint_nose_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y



    keypoint_lefteye_x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE_OUTER].x
    keypoint_lefteye_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE_OUTER].y



    keypoint_righteye_x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE_OUTER].x
    keypoint_righteye_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE_OUTER].y

    dis1 = math.sqrt(pow(keypoint_righteye_x-keypoint_nose_x,2)+pow(keypoint_righteye_y-keypoint_nose_y,2))
    dis2 = math.sqrt(pow(keypoint_lefteye_x-keypoint_nose_x,2)+pow(keypoint_lefteye_y-keypoint_nose_y,2))
    dis3 = math.sqrt(pow(keypoint_righteye_x-keypoint_lefteye_x,2)+pow(keypoint_righteye_y-keypoint_lefteye_y,2))

    #print (dis1, dis2, dis3)
    if dis1<0.05:
        print("looking right")

    elif dis2<0.05:
        print("looking left")

    else:
        print("looking straight")

    keypoint_rightindex_x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX].x
    keypoint_rightindex_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX].y

    dis4 = math.sqrt(pow(keypoint_righteye_x-keypoint_rightindex_x,2)+pow(keypoint_righteye_y-keypoint_lefteye_y,2))
    print(dis4)
    if dis4 < 0.11:
        print("talking on phone")
    else:
        print("not using phone")

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    # B: count how many frames the user seems to be going to nap (half closed eyes)
    drowsy_frames = 0

    # C: max height of each eye
    max_left = 0

    max_right = 0
    while True:

        # get every frame from the web-cam
        ret, frame = cap.read()
        if not ret:
            break

        # Get the current frame and collect the image information
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        # D: collect the mediapipe results
        resultz = face_mesh.process(rgb_frame)

        # E: if mediapipe was able to find any landmanrks in the frame...
        if resultz.multi_face_landmarks:

            # F: collect all [x,y] pairs of all facial landamarks
            all_landmarks = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in resultz.multi_face_landmarks[0].landmark])

            # G: right and left eye landmarks
            right_eye = all_landmarks[RIGHT_EYE]
            left_eye = all_landmarks[LEFT_EYE]

            # H: draw only landmarks of the eyes over the image
            cv2.polylines(frame, [left_eye], True, (0,255,0), 1, cv2.LINE_AA)
            cv2.polylines(frame, [right_eye], True, (0,255,0), 1, cv2.LINE_AA)

            # I: estimate eye-height for each eye
            len_left = open_len(right_eye)
            len_right = open_len(left_eye)
            print(len_right,len_left)

            # J: keep highest distance of eye-height for each eye
            if len_left > max_left:
                max_left = len_left

            if len_right > max_right:
                max_right = len_right

            # print on screen the eye-height for each eye
            cv2.putText(img=frame, text='Max: ' + str(max_left)  + ' Left Eye: ' + str(len_left), fontFace=0, org=(10, 30), fontScale=0.5, color=(0, 255, 0))
            cv2.putText(img=frame, text='Max: ' + str(max_right)  + ' Right Eye: ' + str(len_right), fontFace=0, org=(10, 50), fontScale=0.5, color=(0, 255, 0))

            # K: condition: if eyes are half-open the count.
            if (len_left <= int(max_left / 2) + 1 and len_right <= int(max_right / 2) + 1):
                drowsy_frames += 1
            else:
                drowsy_frames = 0

            # L: if count is above k, that means the person has drowsy eyes for more than k frames.
            if (drowsy_frames > 20):
                cv2.putText(img=frame, text='ALERT', fontFace=0, org=(200, 300), fontScale=3, color=(0, 255, 0), thickness = 3)


        cv2.imshow('img', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
