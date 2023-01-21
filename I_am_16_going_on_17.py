import cv2
import mediapipe as mp
import math
import numpy as np
import time


handy =0
def distance(p1, p2):
    ''' Calculate distance between two points
    :param p1: First Point
    :param p2: Second Point
    :return: Euclidean distance between the points. (Using only the x and y coordinates).
    '''
    return (((p1[:2] - p2[:2]) ** 2).sum()) ** 0.5


def eye_aspect_ratio(landmarks, eye):
    ''' Calculate the ratio of the eye length to eye width.
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :param eye: List containing positions which correspond to the eye
    :return: Eye aspect ratio value
    '''
    N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    return (N1 + N2 + N3) / (3 * D)


def eye_feature(landmarks):
    ''' Calculate the eye feature as the average of the eye aspect ratio for the two eyes
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Eye feature value
    '''
    return (eye_aspect_ratio(landmarks, left_eye) + \
            eye_aspect_ratio(landmarks, right_eye)) / 2


def mouth_feature(landmarks):
    ''' Calculate mouth feature as the ratio of the mouth length to mouth width
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Mouth feature value
    '''
    N1 = distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
    N2 = distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
    N3 = distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
    D = distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
    return (N1 + N2 + N3) / (3 * D)



def run_face_mp(image):
    global handy
    ''' Get face landmarks using the FaceMesh MediaPipe model.
    Calculate facial features using the landmarks.
    :param image: Image for which to get the face landmarks
    :return: Feature 1 (Eye), Feature 2 (Mouth), Feature 3 (Pupil), \
        Feature 4 (Combined eye and mouth feature), image with mesh drawings
    '''

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image_width, image_height = image.shape[1], image.shape[0]
    image.flags.writeable = False
    results = holistic.process(image)


    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )

    # 2. Right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )

    # 3. Left Hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )


    # 4. Pose Detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


    if results.face_landmarks :
        keypoint0 = results.face_landmarks.landmark[130]
        x = int(keypoint0.x * image_width)
        # print(x)

        y = int(keypoint0.y * image_height)

        # keypoint extraction nose

        keypoint1 = results.face_landmarks.landmark[2]
        x1 = int(keypoint1.x * image_width)

        y1 = int(keypoint1.y * image_height)

        # keypoint extraction

        keypoint2 = results.face_landmarks.landmark[359]
        x2 = int(keypoint2.x * image_width)

        y2 = int(keypoint2.y * image_height)



        image = cv2.circle(image, (x, y), radius=3, color=(0, 0, 255), thickness=6)
        image = cv2.circle(image, (x1, y1), radius=3, color=(0, 255, 0), thickness=6)
        image = cv2.circle(image, (x2, y2), radius=3, color=(255, 0, 0), thickness=6)
        distance1 = math.sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))
        distance2 = math.sqrt(pow((x2 - x), 2) + pow((y2 - y), 2))
        distance3 = math.sqrt(pow((x1 - x), 2) + pow((y1 - y), 2))
        # print("left_nose", distance2, "right_nose", distance1)

        cv2.line(image, (int(x), int(y)), (int(x1), int(y1)), (0, 255, 0),
                 thickness=3,
                 lineType=8)
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0),
                 thickness=3,
                 lineType=8)
        cv2.line(image, (int(x2), int(y2)), (int(x), int(y)), (0, 0, 255),
                 thickness=3,
                 lineType=8)

        landmarks_positions = []
        # assume that only face is present in the image
        for _, data_point in enumerate(results.face_landmarks.landmark):
            landmarks_positions.append(
                [data_point.x, data_point.y, data_point.z])  # saving normalized landmark positions
        landmarks_positions = np.array(landmarks_positions)
        landmarks_positions[:, 0] *= image.shape[1]
        landmarks_positions[:, 1] *= image.shape[0]

        ear = eye_feature(landmarks_positions)
        mar = mouth_feature(landmarks_positions)
        if results.left_hand_landmarks:
            try:
                keypoint10 = results.left_hand_landmarks.landmark[12]
                g1 = int(keypoint10.y * image_height)
                handy = g1
                #print("its a g1", g1)
            except:
                handy=0
                #print("no hand")




    else:
        ear = -1000
        mar = -1000

        x1 = 0
        y1 = 0

    return ear, mar, image, x1, y1


def calibrate(calib_frame_count=25):
    ''' Perform clibration. Get features for the neutral position.
    :param calib_frame_count: Image frames for which calibration is performed. Default Vale of 25.
    :return: Normalization Values for feature 1, Normalization Values for feature 2, \
        Normalization Values for feature 3, Normalization Values for feature 4
    '''
    ears = []
    mars = []

    cap = cv2.VideoCapture(0)
    time.sleep(1)
    #print("i")
    while cap.isOpened():
        success, image = cap.read()
        #print("j")
        if not success:
            print("Ignoring empty camera frame.")
            continue

        ear, mar, image,distance1,distance2 = run_face_mp(image)
        if ear != -1000:
            ears.append(ear)
            mars.append(mar)


        cv2.putText(image, "Calibration", (int(0.02 * image.shape[1]), int(0.14 * image.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        cv2.imshow('MediaPipe FaceMesh', image)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
        if len(ears) >= calib_frame_count:
            break

    cv2.destroyAllWindows()
    cap.release()
    ears = np.array(ears)
    mars = np.array(mars)

    return [ears.mean(), ears.std()], [mars.mean(), mars.std()], distance1,distance2


def infer(ears_norm, mars_norm, distance1cal,distance2cal):
    ''' Perform inference.
    :param ears_norm: Normalization values for eye feature
    :param mars_norm: Normalization values for mouth feature

    '''
    ear_main = 0
    mar_main = 0

    decay = 0.9  # use decay to smoothen the noise in feature values

    label = None
    label1 = None

    input_data = []
    frame_before_run = 0

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        global handy
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        ear, mar, image,distance1,distance2 = run_face_mp(image)
        if ear != -1000:
            ear = (ear - ears_norm[0]) / ears_norm[1]
            mar = (mar - mars_norm[0]) / mars_norm[1]

            if ear_main == -1000:
                ear_main = ear
                mar_main = mar

            else:
                ear_main = ear_main * decay + (1 - decay) * ear
                mar_main = mar_main * decay + (1 - decay) * mar

        else:
            ear_main = -1000
            mar_main = -1000
            puc_main = -1000
            moe_main = -1000
        #print("caliberation", distance1cal)
        #print("caliberation2", distance2cal)
        #print("real",distance1)
        #print("dista", distance2)

        if len(input_data) == 20:
            input_data.pop(0)
        input_data.append([ear_main, mar_main])

        frame_before_run += 1
        if frame_before_run >= 15 and len(input_data) == 20:
            frame_before_run = 0
            print("phone",handy)



            if ear_main < -2.0:
                label = "lethargic"
                '''
                ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
                ser.reset_input_buffer()
                ser.write(b"ss")
                time.sleep(5)'''
                print(label)


            elif distance1 < distance1cal-50:
                label="looking_Left"
                print(label)
            elif distance1 > distance1cal+50:
                label="looking_right"
                print(label)
            elif handy <= 300 and handy >0 :

                label = "Talking on Phone"
                handy = 0
                print(label)


            else:
                label = "Neutral"
                print(label)

            print('got label ', label)
        cv2.imshow('MediaPipe FaceMesh', image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()

'''
        cv2.putText(image, "EAR: %.2f" % (ear_main), (int(0.02 * image.shape[1]), int(0.07 * image.shape[0])),
                    cv2.F#ONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(image, "MAR: %.2f" % (mar_main), (int(0.27 * image.shape[1]), int(0.07 * image.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(image, "PUC: %.2f" % (puc_main), (int(0.52 * image.shape[1]), int(0.07 * image.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(image, "MOE: %.2f" % (moe_main), (int(0.77 * image.shape[1]), int(0.07 * image.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    
           '''         '''
        if label is not None:
            if label == 0:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.putText(image, "%s" % (label), (int(0.02 * image.shape[1]), int(0.2 * image.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
                        '''







right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]]  # right eye landmark positions
left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]]  # left eye landmark positions
mouth = [[61, 291], [39, 181], [0, 17], [269, 405]]  # mouth landmark coordinates
states = ['alert', 'drowsy']

# Declaring FaceMesh model

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.8)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=4)

print('Starting calibration. Please be in neutral state')

ears_norm, mars_norm,distance1,distance2 = calibrate()

#print(distance1,distance2)

print('Starting main application')
time.sleep(1)
infer(ears_norm, mars_norm, distance1,distance2)
