import cv2, os
from fer2013_model_lib import FaceRecognition, sub_display, CNN_model
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pwd = 'CNN_model2.h5'
cap = cv2.VideoCapture(0)
while True:
    #Capture images
    ret, frame = cap.read()
    cam, face_gray, is_face = FaceRecognition(frame)
    #Implement CNN
    CAM, buf = CNN_model(pwd, face_gray, is_face)
    #Display
    fig = sub_display(cam,face_gray,CAM,buf,is_face)
    #Click q to close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        plt.close(fig)
        break

#Release camera
cap.release()