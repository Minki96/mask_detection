# import plaidml.keras
# plaidml.keras.install_backend()
import time
import datetime
import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from requests import request
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from pyimagesearch.centroidtracker import CentroidTracker

# from cloud_messaging import mains
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
now = datetime.datetime.now()
facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
model = load_model('models/fourB_detect_mask.model')
ct = CentroidTracker()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print(f'Video시작: {now.year}년 {now.month}월 {now.day}일 {now.hour}시 {now.minute}분 {now.second}초')

ret, img = cap.read()
s_time = time.time()
Id_list = []

while True:
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if not ret:
        break
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(230, 230), mean=(104., 177., 123.))
    facenet.setInput(blob)
    dets = facenet.forward()
    result_img = img.copy()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    rects = []

    for i in range(dets.shape[2]):
        confidence = dets[0, 0, i, 2]
        if confidence < 0.55:
            continue
        else:
            x1 , y1, x2, y2 = int(dets[0, 0, i, 3] * w), int(dets[0, 0, i, 4] * h), int(dets[0, 0, i, 5] * w), int(dets[0, 0, i, 6] * h)
            face = img[y1:y2, x1:x2]
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            rects.append(box.astype("int"))
            try:
                face_input = cv2.resize(face, dsize=(224, 224))
                face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
                face_input = preprocess_input(face_input) 
                face_input = np.expand_dims(face_input, axis=0)
                mask, no_mask = model.predict(face_input).squeeze()

                if mask > no_mask:
                    color = (0, 255, 0)
                    label = 'Mask %d%%' % (mask * 100)

                elif no_mask > 0.6:
                    color = (0, 0, 255)
                    label = 'No Mask %d%%' % (no_mask * 100)

            except:
                continue
        cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
        cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color= color , thickness=2, lineType=cv2.LINE_AA)

    people_count = []
    objects = ct.update(rects)
    try:
        for (objectID, centroid) in objects.items():
            text = "ID{}".format(objectID)
            if text not in Id_list:
                Id_list.append(text)
                if color == (0, 0, 255):
                    print('request..', text)
                    cap_img = cv2.imwrite('./no_mask_person/no_mask_person' + text +'.jpg', result_img)
                    # request.post('http://3.34.183.253:5000/nomask')


            cv2.putText(result_img, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)
            cv2.circle(result_img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            people_count.append(objectID)
    except:
        continue
    cv2.imshow('result', result_img)
    if cv2.waitKey(1) == ord('q'): 
        cv2.destroyAllWindows()
        break

# people_cnt = people_count.pop() + 1
# print('인원수:', people_cnt, '명')
cap.release()
e_time = time.time()
now = datetime.datetime.now()
print(f'Video종료: {now.year} {now.month}월 {now.day}일 {now.hour}시 {now.minute}분 {now.second}초')
print('Video 작동시간:', e_time - s_time)