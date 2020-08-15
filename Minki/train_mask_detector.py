import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend" # plaidML 설정

# argumentparser를 생성하고 arguments를 분석한다.
# 파이썬을 실행파일로 만들 때 외부에서 argument를 받을 경우가 생길 때 argparse를 이용하여 편하게 처리한다.
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="./dataset/",
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="fourB_detect_mask.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# learning_rate(러닝 레이트를 초기화.)
learning_rate = 1e-4
epochs = 20
batch_size = 32

# with_mask 와 without_mask 를 가져와서 imagePaths에 넣고 data 와 이미지들의 class를 초기화 한다.
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# for문을 이용해서 이미지를 가져와서 이미지를 로드, array형태로 변환하고 이미지 전처리 시행.
for imagePath in imagePaths:
	# 파일 이름 추출
	label = imagePath.split(os.path.sep)[-2]

	# 이미지를 224 X 224 사이즈로 주고 전처리하기
	image = load_img(imagePath, target_size=(224, 224))
	# 이제 image에는 0~255사이의 값을 가진 array 형태로 구성되어 있음.
	image = img_to_array(image)

	# 이미지가 mobilenet모델이 요구하는 형식에 적합하도록 변형한다.
	image = preprocess_input(image)

	# data(array형태의 이미지들),labels 리스트(이미지들의 이름들)에 append 한다.
	data.append(image)
	labels.append(label)

# 값을 정규화하기 위해 float32로 type을 바꾸고 array 형태로 변환한다.
data = np.array(data, dtype="float32")
labels = np.array(labels)

# label들의 one-hot encoding을 시행한다(with_mask:0, without_mask:1)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# train과 test set로 나누기.
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# ImageDataGenerator은 이미지 증식.
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# model 로드
baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)

# 합성곱층과 pooling층은 2차원을 다루는데 Dense()층에서 사용하기 위해 Flatten()을 사용해서 1차원으로 변경해줘야 한다.
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)

# Dropout: 모델의 노드를 랜덤하게 끄는 방법으로 과적합을 방지.
# 과적합: 현재 학습하고 있는 데이터 셋은 잘 맞추지만, 훈련 데이터 셋 이외의 새로운 데이터 셋이 들어온다면 모델의 성능이 많이 떨어지는 현상.
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# fully_connected model
model = Model(inputs=baseModel.input, outputs=headModel)

# MobileNet의 layer갯수만큼 for문을 돌면서
# 모델을 컴파일 하기 전에 파라미터가 변하지 않기(=학습되지 않기)를 원하는 layer의 속성을 False로 하고 compile하면 이 속성이 고정됨.
for layer in baseModel.layers:
	layer.trainable = False

# 모델 컴파일
opt = Adam(lr=learning_rate, decay=learning_rate / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# 모델 훈련.
H = model.fit(
	aug.flow(X_train,y_train, batch_size=batch_size),
	steps_per_epoch=len(X_train) // batch_size,
	validation_data=(X_test,y_test),
	validation_steps=len(X_test) // batch_size,
	epochs=epochs)

# 모델 예측
prediction = model.predict(X_test, batch_size=batch_size)

# numpy.argmax: 다차원 배열의 경우에 차원에 따라 가장 큰 값의 인덱스들을 반환해주는 함수.
# X_test의 인덱스 중 가장 큰 값의 인덱스들을 반환.
prediction = np.argmax(prediction, axis=1)

# test의 레이블을 분류 리포트 나타내기.
print(classification_report(y_test.argmax(axis=1), prediction, target_names=lb.classes_))

# 모델 저장.
model.save(args["model"], save_format="h5")

# loss와 accuracy를 나타내서 저장한다.
N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])