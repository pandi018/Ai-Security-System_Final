import numpy
import numpy as np
import pickle
import os
import cv2
import time
import datetime
import pandas as pd
import imutils

curr_path = os.getcwd()
print("Loading face detection model")
proto_path = os.path.join(curr_path, 'model', 'deploy.prototxt')
model_path = os.path.join(curr_path, 'model', 'res10_300x300_ssd_iter_140000.caffemodel')
face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

print("Loading face recognition model")
recognition_model = os.path.join(curr_path, 'model', 'openface_nn4.small2.v1.t7')
face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

names = []
for i in os.listdir("C:/Users/YOGAPRIYA/PycharmProjects/Face/local_disk"):
    name= i.replace("////", "//")
    names.append(os.path.abspath(name))
recognizer = pickle.loads(open('recognizer.pickle', "rb").read())
le = pickle.loads(open('le.pickle', "rb").read())
print("Starting test video file")
path = "C:/Users/YOGAPRIYA/PycharmProjects/Face/local_disk"
# no videos are there
dir_list = os.listdir(path)
final_path = []
for i in range(0,len(dir_list)):
    new = path + "/" + dir_list[i]
    #print(new)
    #vs=cv2.VideoCapture(0)
    vs = cv2.VideoCapture(new)
    text1 = []
    text2=[]
    count=0
    time.sleep(1)
    while True:
        ret, frame = vs.read()
        width_new= vs.get(3)
        width_new=int(width_new)
        frame = imutils.resize(frame, 1500)
        (h, w) = frame.shape[:2]
        image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        face_detector.setInput(image_blob)
        face_detections = face_detector.forward()

        for i in range(0, face_detections.shape[2]):

            '''here we taken only the face which one have accuracy 50% and convert the imageblob which is stored in the face_detections
                   after the encoding process we are going to blob the image by resizeing it into 96,96 image and store the image in the 
                   face_recognizer to recognize the face'''

            confidence = face_detections[0, 0, i, 2]

            if confidence >= 0.5:
                box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]

                (fH, fW) = face.shape[:2]

                face_blob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), True, False)

                face_recognizer.setInput(face_blob)
                vec = face_recognizer.forward()

                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                text1.append("{:.2f}".format(proba * 100))
                # print(text1)
                if(len(text1)>0):
                    final_path.append(new)

                count1=" "

                text ="{}".format(name)
                text2.append(text)
                # print(text2)
                count=count+1
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                if(text=="adam"):
                    cv2.putText(frame, text, (startX-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    # cv2.putText(frame, str(count) , (startX+15, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (85, 96, 237), 2)
                    # print("Class A adam 18_07_1980")
                elif(text=="mary"):
                    cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    # cv2.putText(frame, str(count) , (startX+15, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (85, 96, 237), 2)
                    # print("Class B mary21_03_2000")
                elif(text=="mathew"):
                    cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    # cv2.putText(frame, str(count) , (startX+15, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (85, 96, 237), 2)

                    # print("Class B mathew20_6_1990")
                else:
                    sum1=1
                    # cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    # print("Class C Other Customer")
        frame = cv2.resize(frame, (500, 500))
        cv2.imshow("Frame", frame)
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
#for i in files:
    #print("Absolute path:",os.path(i))
print(set(final_path))
cv2.destroyAllWindows()



