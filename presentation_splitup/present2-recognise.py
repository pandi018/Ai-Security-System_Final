import numpy
import numpy as np
import pickle
import os
import cv2
import time
import datetime
import pandas as pd
import imutils
import pywhatkit
import winsound
frequency=2500
duration=100

curr_path = os.getcwd()

print("Loading face detection model")
proto_path = os.path.join(curr_path, 'model', 'deploy.prototxt')
model_path = os.path.join(curr_path, 'model', 'res10_300x300_ssd_iter_140000.caffemodel')
face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

print("Loading face recognition model")
recognition_model = os.path.join(curr_path, 'model', 'openface_nn4.small2.v1.t7')
face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

recognizer = pickle.loads(open('recognizer.pickle', "rb").read())
le = pickle.loads(open('le.pickle', "rb").read())
print("Starting test video file")
# vs = cv2.VideoCapture("testvideo.mp4")
vs=cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
ret, frame = vs.read()
fgmask = fgbg.apply(frame)
kernel = np.ones((5,5), np.uint8)
text1 = []
text2=[]
count=0
time.sleep(1)
while True:

    ret, frame = vs.read()
    frame = imutils.resize(frame, width=1500)
    # code for tampering of camera
    a = 0
    bounding_rect = []
    fgmask = fgbg.apply(frame)
    fgmask = cv2.erode(fgmask, kernel, iterations=5)
    fgmask = cv2.dilate(fgmask, kernel, iterations=5)
    contours, hierachy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        bounding_rect.append(cv2.boundingRect(contours[i]))
    for i in range(0, len(contours)):
        if bounding_rect[i][2] >= 40 or bounding_rect[i][3] >= 40:
            a = a + (bounding_rect[i][2]) * bounding_rect[i][3]
        if (a >=
                int(frame.shape[0]) * int(frame.shape[1]) / 3):
            cv2.putText(frame, "TAMPERING DETECTED", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            # pywhatkit.sendwhatmsg("+91 6383997876","jsut a check",5,30)
            year, month, day, hour, min = map(int, time.strftime("%Y %m %d %H %M").split())
            print(hour,min)
            pywhatkit.sendwhatmsg("+91 6383997876","EMERGENCY TAMPERING DETECTED",hour,min+1)


    (h, w) = frame.shape[:2]

    image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (108.0, 190.0, 140.0), True, True)
    face_detector.setInput(image_blob)
    face_detections = face_detector.forward()

    for i in range(0, face_detections.shape[2]):

        '''here we taken only the face which one have accuracy 50% and convert the imageblob which is stored in the face_detections
               after the encoding process we are going to blob the image by resizeing it into 96,96 image and store the image in the 
               face_recognizer to recognize the face'''

        confidence = face_detections[0, 0, i, 2]

        if confidence >= 0.9:
            box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]

            (fH, fW) = face.shape[:2]

            face_blob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), True, True)

            face_recognizer.setInput(face_blob)
            vec = face_recognizer.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            text1.append("{:.2f}".format(proba * 100))
            print(text1)
            count1=" "

            text ="{}".format(name)
            text2.append(text)
            print(text2)
            count=count+1
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            if(text=="adam"):
                cv2.putText(frame, text, (startX-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # cv2.putText(frame, str(count) , (startX+15, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (85, 96, 237), 2)
                print("Class A adam 18_07_1980")
            elif(text=="mary"):
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # cv2.putText(frame, str(count) , (startX+15, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (85, 96, 237), 2)
                print("Class B mary21_03_2000")
            elif(text=="mathew"):
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # cv2.putText(frame, str(count) , (startX+15, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (85, 96, 237), 2)

                print("Class B mathew20_6_1990")
            else:
                # cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print("Class C Other Customer")
    re_frame=cv2.resize(frame,(500,500))
    re_frame1=cv2.resize(fgmask,(500,500))
    cv2.imshow("alter_Frame",re_frame1)
    cv2.imshow("Frame", re_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# text2=np.array(text2)
# df1=pd.DataFrame(text2)
# df1.to_csv("Name_final_.csv",mode='a', index=False)
#
# text1=np.array(text1)
# df = pd.DataFrame(text1)
# df.to_csv('accuracy_final_.csv', mode='a', index=False)
cv2.destroyAllWindows()



