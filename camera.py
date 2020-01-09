from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import tensorflow as tf
global graph,model
graph = tf.get_default_graph()

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import time
from datetime import datetime

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]

#feelings_faces = []
#for index, emotion in enumerate(EMOTIONS):
   # feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

# starting video streaming
def Emo():
    #camera = cv2.VideoCapture('C:/Users/1011696/Documents/Python_Scripts/Emo/FaceDetection/static/video.avi')
    camera = cv2.VideoCapture(0)
    df = pd.DataFrame(columns=['Time','Emotion'])
    start_time = datetime.now()
    t0 = time.time()

    while True:
        frame = camera.read()[1]
        t1 = time.time() # current time
        num_seconds = t1 - t0 # diff
        global preds,label,fX, fY, fW, fH

        if num_seconds > 30:  # e.g. break after 30 seconds
                break

        #reading the frame
        frame = imutils.resize(frame,width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = frame.copy()
        if len(faces) > 0:
            faces = sorted(faces, reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
                        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
                # the ROI for classification via the CNN
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            with graph.as_default():
                preds = emotion_classifier.predict(roi)[0]
            
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            
            end_time = datetime.now()
            df = df.append({'Time':(end_time - start_time),'Emotion':label}, ignore_index=True)

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                    # construct the label text
                    text = "{}: {:.2f}%".format(emotion, prob * 100)

                    # draw the label + probability bar on the canvas
                   # emoji_face = feelings_faces[np.argmax(preds)]

                    
                    w = int(prob * 300)
                    cv2.rectangle(canvas, (7, (i * 35) + 5),
                    (w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
                    cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                                  (0, 0, 255), 2)

                    #print('{}  {}'.format(label,(end_time - start_time)))
    #    for c in range(0, 3):
    #        frame[200:320, 10:130, c] = emoji_face[:, :, c] * \
    #        (emoji_face[:, :, 3] / 255.0) + frame[200:320,
    #        10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)


        #cv2.imshow('your_face', frameClone)
        #cv2.imshow("Probabilities", canvas)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
    print(df)
    #df['Emotion'].value_counts().plot('pie').invert_yaxis()
    #plt.savefig("C:/Users/1011696/Documents/Python_Scripts/Emo/FaceDetection/static/people_photo/pie_output.png")
    #df['Emotion'].value_counts().plot('bar')
    #plt.savefig("C:/Users/1011696/Documents/Python_Scripts/Emo/FaceDetection/static/people_photo/bar_output.png")
    sns.set_style("dark")
    sns.countplot(df.Emotion)
#    plt.savefig("C:/Users/1011696/Documents/Python_Scripts/login2/FaceDetection/static/people_photo/bar_output.png")   
    plt.savefig("C:/Users/1011696/Documents/Python_Scripts/Face/login2 - Copy(V1)/FaceDetection/static/people_photo/bar_output.png") 
    df=[]
    camera.release()
    
    