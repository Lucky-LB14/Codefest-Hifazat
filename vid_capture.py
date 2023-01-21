import cv2  #importing the library


vs = cv2.VideoCapture(0) #definind the camera type

while True:
    ret, frame = vs.read()
    if not ret:
        continue
    # Processing of image and other stuff here
    frame = cv2.resize(frame, (800, 500),
                       interpolation=cv2.INTER_NEAREST)
    cv2.imshow('CircuitView', frame)        #quiting the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break