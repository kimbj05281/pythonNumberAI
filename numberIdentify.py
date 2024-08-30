import tensorflow as tf
import keras
from keras import Sequential #케라스 모델도구 중 시퀸셜 모델 함수
from keras import layers
from keras import utils
from keras import datasets
Dense = layers.Dense #layers 도구 중 Dense(전결합층) 도구 불러옴
Activation = layers.Activation #layers 도구 중 Activation 도구 불러옴
to_categorical = utils.to_categorical #유틸 도구 중 to_categorical 함수 불러옴(원-핫 인코딩 구현가능)
mnist = datasets.mnist
import numpy as np
import matplotlib.pyplot as plt #시각화

(x_train, y_train), (x_test, y_test) = mnist.load_data() #mnist 데이터셋에서 데이터 가져오기
print("x_train shape", x_train.shape) #28x28 픽셀의 사진(학습데이터)이 60000개가 있다
print("y_train shape", y_train.shape) #정답 60000개가 들어있음
print("x_test shape", x_test.shape) #문제 10000개 
print("y_test shape", x_test.shape) #정답 10000개

X_train = x_train.reshape(60000,784) #1차원 배열로 reshape 하기
X_test = x_test.reshape(10000,784) #위와 동일
X_train = X_train.astype('float32') #정규화하기 위해 데이터를 0~1 사이의 값으로 바꿈. 하지만 정수형이기 때문에 실수형으로 변경
X_test = X_test.astype('float32') #정규화하기 위해 데이터를 0~1 사이의 값으로 바꿈. 하지만 정수형이기 때문에 실수형으로 변경
X_train /= 255
X_test /= 255
print("X Training matrix shape", X_train.shape)
print("X Testing matrix shape", X_test.shape)

Y_train = to_categorical(y_train, 10) #수치형 데이타를 범주형 데이터를 만들어주는 함수. 원 핫(하나만 강조) 0~9숫자중에 고르기 때문에 10. (60000, 10)
Y_test = to_categorical(y_test, 10) # shape 이 (10000,)에서 (10000, 10)으로 바뀜

# 인공지능 모델 설계
# 4개의 층으로 이루어진 모델 만들기
# 첫 번째 층 : 입력층, 데이터를 넣는 곳(뉴런 784개. 이유는 784개의 데이터가 한 줄로 이루어져있기 때문)
# 두 번째, 세 번째 층 : 은닉층(두 번쨰 층 뉴런 : 512개, 세 번째 층 뉴런 : 256개. 활성화 함수로는 relu 함수 사용)
# 네 번째 층 : 출력층, 결과 출력(뉴런 10개. 입력된 이미지를 10개로 구분하기 위해)
# 가장 높은 확률의 값으로 분류하기 위해 softmax 함수 사용

model = Sequential()
model.add(Dense(512, input_shape=(784,))) #모델에 층을 추가(add)
model.add(Activation('relu')) #다음 층으로 값을 전달할 떄 relu 함수 사용
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

#모델 학습 시키기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#compile 함수 규착1 : 오차값 계산 방법 알려줘야 함. 이 인공지능은 이미지를 10개 중 하나로 분류하므로 다중분류 문제에 해당
# --> 따라서 categorical_crossentropy 방법 사용
#compile 함수 규칙2 : 오차를 줄이는 방법을 알려줘야 함(경사하강법). 오차를 줄이기 위해 optimizer 사용. 여기에서 adam 방법 사용
#compile 함수 규칙3 : 학습 결과를 어떻게 확인할지 알려줘야 함. 여기선 정확도로 모델학습 결과를 확인. 예측값과 실제값의 비교를 통해 정답 비율 알려줌

model.fit(X_train, Y_train, batch_size=128, epochs=10, verbose=1)

score = model.evaluate(X_test, Y_test) #evaluate : 모델의 정확도 평가기능 제공. 문제와 정답을 파라미터로 입력
print("Test loss:",score[0]) #score 변수에는 오차값과 정확도가 들어있음 
print("Test score", score[1]) 

#모델의 학습 결과 확인
predicted_classes = np.argmax(model.predict(X_test), axis=1) #argmax함수에서 열 중에서 혹은 행 중에서 가장 큰 것을 고를지 알려주는 axis. 0은 세로 1은 가로
correct_indices = np.nonzero(predicted_classes == y_test)[0] #nonzero는 0이 아닌 값, 즉 일치하는 값이 어디인지 찾는 것
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

print(correct_indices)
print(incorrect_indices)