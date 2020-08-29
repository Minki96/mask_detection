# Mask Detection

### 프로젝트 : 코로나 확산 방지를 위한 마스크 식별 출입관리 시스템
* Caffe-model을 이용한 얼굴인식
* MobileNet V2 모델 사용
* data set = mask image 1838 + no mask image 1995
* Tensorflow, keras, opencv, numpy 라이브러리 사용
* face tracking 기능으로 얼굴마다 ID부여
* 얼굴별 면적을 계산하여, 특정 거리에서만 분석.
* 앱과 연동하여 실시간 알람   
<img src="/Minki/cap.jpeg" width="250px" height="300px" title="px(픽셀) 크기 설정" alt="cap"></img><br/>    

참고 : https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/
