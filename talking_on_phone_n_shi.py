import cv2
import mediapipe as mp
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)


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
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
