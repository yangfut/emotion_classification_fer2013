from keras.models import Model
from matplotlib import pyplot as plt
import cv2, os
import numpy as np
import tensorflow as tf

def FaceRecognition(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find faces
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=18,minSize=(48, 48),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) != 0:
        is_face = 1
        for (x, y, w, h) in faces:
            face = np.round(gray[y:y + h, x:x + w].copy() / 255, 8)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        is_face = 0
        face = np.round(gray.copy() /255,8)
    return img, face, is_face

def sub_display(cam,face_gray,CAM,buf,is_face):
    y1,y2,x1,x2 = 50,650,280,920
    cam = np.copy(cam[y1:y2, x1:x2])
    b, g, r = cv2.split(cam)
    camm = cv2.merge([r, g, b])
    height, width, depth = cam.shape
    if is_face == 1:
        face_resized = cv2.resize(face_gray, (width, height))
        CAM_resized = cv2.resize(CAM, (width, height))
    elif is_face == 0:
        face_resized = cv2.cvtColor(camm, cv2.COLOR_BGR2GRAY)
        CAM_resized = np.zeros((height,width))

    fig = plt.figure('webcam', figsize=(13, 5))
    plt.suptitle('Emtion Classification Through Deep Learning', fontsize=22)
    class_str = '7 categories: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral'
    sub1 = 'Webcam image'
    sub2 = 'Cropped face'
    sub3 = 'Inside neural network'
    author = '@author: Daniel'
    plt.figtext(.01, .95, author, fontsize=12)

    plt.figtext(.24, .87, class_str, fontsize=14, style='italic')
    plt.figtext(.12, .03, sub1, fontsize=12)
    plt.figtext(.46, .03, sub2, fontsize=12)
    plt.figtext(.765, .03, sub3, fontsize=12)
    plt.subplot(1, 3, 1), plt.imshow(camm, cmap='gray'), plt.axis('off')
    plt.subplot(1, 3, 2), plt.imshow(face_resized, cmap='gray'), plt.axis('off')
    plt.figtext(.755, .85, buf, fontsize=18, color='b')
    plt.subplot(1, 3, 3), plt.imshow(face_resized, alpha=0.8), plt.axis('off')
    plt.imshow(CAM_resized, cmap='jet', alpha=0.5)

    plt.tight_layout()
    plt.pause(0.001)
    plt.clf()
    return fig

def CNN_model(pwd,face_gray, is_face):
    def label(x):
        if x == 0:
            label_str = 'Angry'
        elif x == 1:
            label_str = 'Disgust'
        elif x == 2:
            label_str = 'Fear'
        elif x == 3:
            label_str = 'Happy'
        elif x == 4:
            label_str = 'Sad'
        elif x == 5:
            label_str = 'Surprise'
        elif x == 6:
            label_str = 'Neutral'
        return label_str

    if is_face == 1:
        model = tf.keras.models.load_model(pwd)
        cam_model = Model(inputs=model.input, outputs=(model.layers[-4].output, model.layers[-1].output))
        face = cv2.resize(face_gray,(48,48))
        input = face.reshape(1,48,48,1)
        features, results = cam_model.predict(input)
        one_img = features[0, :, :, :]
        pred = np.argmax(results[0])
        gap_weights = model.layers[-1].get_weights()[0]
        cam_weights = gap_weights[:, pred]
        cam_output = np.dot(one_img, cam_weights)
        output = cv2.resize(cam_output, (48, 48))
        buf = 'P: ' + label(pred) + ', P. Val: ' + str(np.round(results[0][pred],3))
    elif is_face == 0:
        output = 0
        buf = 'No face here'
    return output, buf