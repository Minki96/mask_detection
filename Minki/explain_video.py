import time
import datetime
import requests
import argparse
import face_recognition as rec_face
import imutils
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json
from pyimagesearch.centroidtracker import CentroidTracker
from line import warning

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 모델 불러오기.
# dnn: opencv의 새로운 심층 신경망 모듈: 사전처리 된 딥러닝 모델을 통해 이미지를 사전 처리하고 분류를 준비하는데 사용할 수 있는 두가지 기능이 있음
# cv2.dnn.blobFromImage()
# cv2.dnn.blobFromImages()
# ==> 위 두개는 평균 뺴기, 스케일링, 선택적인 채널 교환
now=datetime.datetime.now()
# face detection모델.
# cv2.dnn.readNet: 네트워크를 메모리에 로드
facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
# mask detection 모델.=> keras모델
ap = argparse.ArgumentParser()

ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability to filter weak detections")
args = vars(ap.parse_args())

model = load_model('models/fourB_detect_mask.model')

ct = CentroidTracker()
# img = cv2.imread('imgs/02.jpg') # 이미지 로드
# h,w = img.shape[:2] # 이미지의 높이와 넓이를 저장
# plt.figure(figsize=(16,10))
# plt.imshow(img[:,:,::-1])# 이미지 확인 #[:,:,::-1]: BGR을 RGB로 바꿔주는 채널

# VideoCapture: real-time object detection( 실시간 객체 탐지)에 유리.
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
s_time=time.time()
print(f'Video시작: {now.year}년 {now.month}월 {now.day}일 {now.hour}시 {now.minute}분 {now.second}초')

ret, img = cap.read()
# 동영상 저장...
store_video = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
print('Video 저장 중.. 종료 후 result.mp4를 확인하세요.')
out = cv2.VideoWriter('result.mp4', store_video, cap.get(cv2.CAP_PROP_FPS), (img.shape[1], img.shape[0]))
Id_list = []

while True:
    ret, img = cap.read()
    if not ret:
        break
    # 이미지 전처리와 추론
    h, w = img.shape[:2]
  #  print(h) ==> 480
  #  print(w) ==> 640
    # blobFromImages: 여러 이미지를 전달할 수 있음.
    # process Image for face detection : 얼굴 감지(찾기..?)를 위한 이미지 처리->brobfromImage(dnn모듈이 사용하는 형태로 이미지를 변형, axis순서만 바뀜) 사용
    blob = cv2.dnn.blobFromImage(img, # 분류를 위해 심층 신경망을 통과하기 전에 전처리하려는 입력 이미지
                                 scalefactor=1., # 평균 뺄셈을 수행한 후 선택적으로 이미지를 스케일링 할 수 있다. 기본값:1.0(스케일링 없음)
                                 size=(300, 300), # cnn의 공간 크기
                                 mean=(104., 177., 123.)) # 평균 빼기 값. RGB평균의 3 튜플일 수 도 있고, 단일 값일 수도 있음. 제공된 모든값이 이미지의 모든 채널에서 `빠짐.
    # print('*****', blob.shape) # (1,3,300,300) ==>[[[[a,b,c,d,,,,,]],[[e,f,g,h,,,,]],[[i,j,k,l,,,,]]] # 3개의 2차원 안에 값들이 들어가 있고 그렇게 []리스트가 총 300개, row 300개가 있다.
    facenet.setInput(blob) # 모델에 인풋 데이터를 넣는다.
    dets = facenet.forward() # 결과를 추론한다(inference result), 얼굴들(?)을 배열로....
    #print(dets)
    # print(f'***** dets.shape={dets.shape}') #(1,1,200,7) ==>[[[안에 [],[],[]형식으로 있음]]], 200,7==> 200개의 행 =>[]X200개 []안에 원소 갯수가 7개
    result_img = img.copy()
    rects = []
    #dets.shape[2] :200
    for i in range(dets.shape[2]): # 여러개의 얼굴이 detection 될 수 있으니 for문을 사용하여 검사. =-> 200은 행의 갯수이고 그행의 갯수를 for문을 돌리면 얼굴이 몇개가 webcam에
        # detection 되고 있는지 잡힘.
        confidence = dets[0, 0, i, 2] # detection 한 결과가 얼마나 자신있는지-->confidence, 200개의 행을 for문을 돌려서 검사해서 얼굴을 detection한다=>200개의 행 2번인덱스의 row..
        if confidence > args['confidence']:
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h]) # 200개의 행에 3번째 부터 (7-1=)6번째 row까지의 값을 리스트 형식 (=[값]) 으로 반환해서 [3번째,4번째,5번째,6번째 row] X [w,h,w,h]를 한다.
            # print(box.shape) # (4,)
            rects.append(box.astype("int"))
            # print(len(rects))==> webcam에 잡히는 얼굴 갯수로 나옴 ,, 나만 인식하고 있으면 1, 2명이 들어오면 2,,
            # print(rects) ==> array([440,244,579,431)=> array([a,b,c,d])=> a=실제 x좌표의값 ,b:실제 y좌표 값, c:x좌표값에 일정 값이 곱해진값, d=y좌표값에 일정값이 곱해진값.
        if confidence < 0.5:
            continue # 0.5미만인 것은 다 넘김.

        # x나 y의 바운딩 박스를 구해줌
        x1 = int(dets[0, 0, i, 3] * w)
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)

        face = img[y1:y2, x1:x2] # 바운딩 박스를 가지고 원본 이미지에서 얼굴만 추출해내기.
        # print(f'***** face.shape={face.shape}')
        # faces.append(face) #얼굴들 저장

        # 얼굴만 잘 저장되었는지 확인하기.
        # for i, face in enumerate(faces):
        # plt.subplot(1,len(faces),i+1)
        # plt.imshow(face[:,:,::-1])

        # detect masks from faces
        # for i,face in enumerate(faces): #마스크를 썻나 안썻나 확인하는 for문(detect mask from faces) 71~79line까지 들어가야함.
        # 71~74까지는 전처리하는 코드
        try:
            face_input = cv2.resize(face, dsize=(224, 224)) #dsize=(224, 224):이미지 크기 변형.
            # print(f'***** face_input.shape={face_input.shape}')
            face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB) #opencv는 기본 BGR로 되어있는데 RGB로 바꿈(mean substraction): 컬러시스템 변경.
            face_input = preprocess_input(face_input) # MobileNetV2에서 하는 preprocessing과 똑같이 하기 위해 preprocess_input을 해줌.
            face_input = np.expand_dims(face_input, axis=0) # 위까지 추가하면 RGB는 (224,224,3)으로 생성되는데 (1,224,224,3)으로 나와야 되서 차원을 추가하는 expand_dims 사용해서 0번 axis에 차원을 하나 추가해줌,.
            # print(f'***** face_input.shape={face_input.shape}')

            mask, nomask = model.predict(face_input).squeeze() # 마스크를 쓴 확률, 안쓴 확률이 나옴.
            # plt.subplot(1,len(faces,i+1))
            # plt.imshow(face[:,:,::-1])
            # plt.title('%2.f%%' % (mask*100))

            # for 문으로 위 79라인 까지 돌려서 마스크를 쓴 확률 안 쓴 확률을 사진에서 나타내본다.

            if mask > nomask:
                color = (0, 255, 0)
                label = 'Mask %d%%' % (mask * 100)
            else:
                color = (0, 0, 255)
                label = 'No Mask %d%%' % (nomask * 100)
        except Exception as e:
            print(f'***** e={e}')
            continue
        # pt1, pt2는 사각형의 위치를 지정해주는 것이고 위에서 구한 값을 그대로 넣어야함, 만약 +50이나 다른 수를 더할 경우 기존위치에서 그만큼 이동하게됨,
        cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
        # org = 출력 문자 시작 위치 좌표( 좌측 하단), fontScale=폰트크기, thickness=폰트두께,
        cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2, lineType=cv2.LINE_AA)

    objects = ct.update(rects)
    people_count=[]
    # print(objects.items()) #odict_items([(0, array([383, 242]))]) 이런식으로 화면에 얼굴들이 계속 비춰지면 계속 프린트 됨.
                                        # 만약 감지 할 얼굴이 없다면 그냥 빈 1차원 array로 뜨고
                                        # 얼굴을 다시 감지해서 아이디가 바뀌면 0은 1이된다
                                        # 즉, odict_items([objectId,centroid([얼굴 좌표]))로 이루어져 있음==> while True동안 계속 출력됨.

    for (objectID, centroid) in objects.items():

        # s_time=time.time()
        text = "ID {}".format(objectID)
        # print(centroid) #  centroid==[a,b] 형태로 이루어져있다
        # 따라서 centroid의 axis값..? 인덱스 값은 0과 1로 이루어져 있음
        # centroid는 얼굴을 감지하는 사각형 (rectangle)의 좌표값을 저장하고 있으며
        # centroid[0]은 얼굴(사각형)의 x좌표 값, 즉 얼굴이 좌우로 움직일 때 x좌표값을 나타내고
        # 좌표값의 최솟값은 대략 99에서 최대값은 대략 516 정도의 사이에 있다(즉 카메라 앵글의 x축의 최소값과 최대값을 나타낸다)
        # centroid[1]은 y값이며 대략 145~340 사이에 있는데 x축과 다른점은 y값의 방향은 위에서 아래로 올때 값이 커진다.. 얼굴을 올리면 y값이 작아지고 얼굴을 내리면 y값이 커짐.
        cv2.putText(result_img, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(result_img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        people_count.append(objectID)
        if color == (0, 0, 255):  # 빨간색 상자이면..
            if text not in Id_list:  # id list에 id가 없다면..
                Id_list.append(text)  # id 리스트에 추가한다. 마스크 안쓴사람의 ID를,,
                print('request..', text)
                warning(text)

    out.write(result_img)
    cv2.imshow('result', result_img)
    if cv2.waitKey(1) == ord('q'):
        break
print('인원수:',people_count[-1:],'명')
# 마스크 미착용자에 대한 warning, 최종인원수...
# 인원수, 아이디 사라지는 시간.
out.release()
cap.release()
e_time=time.time()
now=datetime.datetime.now()
print(f'Video종료: {now.year} {now.month}월 {now.day}일 {now.hour}시 {now.minute}분 {now.second}초')
print('Video 작동시간:',e_time-s_time)

