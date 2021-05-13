# import the necessary packages
import time

from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():

	# OrderedDict() 넣는 순서대로 키를 내어주는 역할을 함.
	# objects: 추가 될 객체들..=> ObjectID
	def __init__(self, maxDisappeared=7):
		self.nextObjectID = 0 # 인수 설정.( 아이디가 올라갈 때 마다 그 아이디의 위치, 즉 인덱스를 반환 해 줄 것임)
		self.objects = OrderedDict() # Id들을 추가할 딕셔너리 생성
		self.disappeared = OrderedDict() # 사라진 ID들을  추가할 딕셔너리 생성
		self.maxDisappeared = maxDisappeared  # 변수 설정.

	def register(self, centroid):
		self.objects[self.nextObjectID] = centroid # OrderedDict의  [nextObjectID] 위치에 centroid([int a,int b]) 좌표를 추가한다.==> 내얼굴의 좌표값을 (centroid에) 넣음.
		self.disappeared[self.nextObjectID] = 0 # disappeared OrderedDict의 nextObjectID위치에 0을 넣는다 ( because this function  is registering )==> 아이디값을 넣음
		self.nextObjectID += 1 # 다음 아이디를 하나 올려준다.

	def deregister(self, objectID):
		del self.objects[objectID] # objects의[objectId] 인덱스 즉 그값을 del(삭제)
		del self.disappeared[objectID]

	def update(self, rects):
		if len(rects) == 0: # 사각형이 생성이 안되었다면.
			for objectID in self.disappeared.keys(): # disappeared의 인덱스들=>dictionary에선 key들..0,1,2,3,4,,,,,,
				self.disappeared[objectID] += 1 # disappeared OrderedDict의 objecId 값의 인덱스의 값을 의미하고 그 값을 1올려준다=> 즉 아이디를 1올려줌.
				if self.disappeared[objectID] > self.maxDisappeared: # 만약 그 위치의 값이 maxDisappeared보다 크다면 Id와 centroid(내얼굴의 좌표값) 삭제한다.
					self.deregister(objectID)
			return self.objects # 좌표값을 리턴한다.

		# 현재 프레임의 위치를 초기화 한다(0으로) if, len(rects)가 2= 즉 사람이 2명 들어와있어서 np.zeors(2,2)라면 ==> [[0 0] [0 0]]이 된다.
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# 사각형의 각각좌표를 도는 for문
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY) # np.zeros인 inputCentroids의 i번째 행에 cX와 cY를 넣는다.

		# id가 아무것도 없다면  좌표의 중점을 이용해 register한다.
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i]) # 사각형 좌표의 중점을 담은 inputCentroid의 i번째 행을 리턴한다.

		# id(objects)를 추적하고 있었다면,
		else:
			# 부여된 아이디값들과 좌표값들이 일치하는 지 비교하기
			# 아이디와 좌표를 리스트형태로 저장한다.
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# 각 objectCentroids와 input centroids 사이의 거리를 구한다
			# 목적은 inputcentroid와 현재의 objectcentroid를 match시키기 위함이다
			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			# 두개를 matching 시키기 위해서는
			# 1.각 행에서 가장 작은 값을 찾는다# 거리의 최소값,,,
			# 2. 가장 작은 값의 행을위해  그들의 최소값에 기초한 행 인덱스들을 정렬한다
			# 각 행의 최소값을 구해서 값이 작은 순서대로 나열한다
			# ex D.min(axis=1)==>[102.85912696  91.67878708  80.49844719  69.3181073   58.13776741]] ==> 인덱스를 매기면  0,1,2,3,4==> 값이 작은 인덱스의 순서대로 = 4,3,2,1,0
			rows = D.min(axis=1).argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			# 각 컬럼에서 제이
			# D.argmin(axis=1) : 각 행의 최소값의 인덱스 값을 리턴한다
			cols = D.argmin(axis=1)[rows]

			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				# val
				if row in usedRows or col in usedCols:
					continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		# return the set of trackable objects
		return self.objects