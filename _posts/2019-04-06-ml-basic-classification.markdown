---
layout: post
title:  "머신러닝 예제 : 기초적인 분류 문제"
subtitle:   "머신러닝 기초"
categories: devlog
tags: machine_learning basic_classification
---
---
이번 포스팅에서는 코딩기초학습의 Hello World가 있는 것처럼 머신러닝의 Hello World를 예제를 통해 다루겠습니다.

이 예제는 운동화나 셔츠, 바지 같은 옷의 간단한 이미지(28 x 28 픽셀)를 분류하는 신경망 모델을 훈련합니다.

완전한 텐서플로우(TensorFlow)프로그램이 어떻게 돌아가는지 빠르게 살피는 학습을 목표로 하고 있습니다.

이 예제는 텐서플로우 모델을 만들고 훈련할 수 있는 고수준 API인 tf.keras를 사용합니다.

파이참을 이용할 것입니다.

이 예제는 텐서플로우 튜토리얼 Basic classification을 바탕으로 학습하는 예제입니다.

---
## 라이브러리 임포트 하기

~~~python
from __future__ import absolute_import, division, print_function, unicode_literals

# tensorflow와 tf.keras를 임포트 사용합니다
import tensorflow as tf
from tensorflow import keras

# 헬퍼(helper) 라이브러리를 임포트합니다
import numpy as np
import matplotlib.pyplot as plt

# tensorflow의 버전 확인
print(tf.__version__)
~~~
### <center>결과</center>

![결과](https://drive.google.com/uc?id=1EXsUjGnJ-2CdqIxp67CaV5pX38YFl5Mq)

우선 이번 예제에서 쓸 라이브러리들을 임포트 해줍니다.
`matplotlib`는 파이썬에서 자료를 ``시각화(visulaization)``하는 패키지입니다.
특히 플롯(그래프)을 그릴 때 주로 많이 쓰이는 2D, 3D 플롯팅 패키지입니다.

---

## 패션 MNIST 데이터셋 임포트하기
다음은 이번 예제에서 기계가 학습할 이미지를 임포트 해줍니다. 10개의 종류와 70000개의 흑백 이미지(28 x 28 픽셀)로 구성된 `패션 MNIST` 를 사용합니다.

![패션 MNIST 이미지](https://drive.google.com/uc?id=1Ja2Yog3ZBvUTccmcJIGycKtw-gSC1FF6)
![패션 MNIST 임베딩](https://drive.google.com/uc?id=13qOWdz9i-xoCdAsAkbZ_WzCtuOs_ZCv-)

네트워크를 훈련하는데 60,000개의 이미지를 사용하고, 네트워크가 얼마나 정확하게 이미지를 분류하는지 10,000개의 이미지로 평가하도록 합니다. `패션 MNIST 데이터셋`은 텐서플로우에서 바로 `임포트하여 적재`할 수 있습니다


~~~python
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
~~~

`패션 mnist`를 `fashion_mnist`에 적재하고 `load_data()`를 호출하여 4개의 넘파이 배열에 반환합니다.

`train_images`, `train_labels`는 모델 학습에 훈련되는 세트이고 `test_images`, `test_labels`는 모델 테스트에 사용되는 테스트 세트입니다.

이미지는 28x28 크기의 넘파이 배열이고 픽셀 값은 0과 255 사이입니다. 레이블(label)은 0에서 9까지의 정수 배열입니다. 이 값은 이미지에 있는 옷의 클래스(class)를 나타냅니다. 간단하게 말하면 레이블은 옷의 종류를 나타내는거죠.

![패션 MNIST 레이블](https://drive.google.com/uc?id=1a5x_Ui__U1jPzifIZWYitHF4EubWVXKu)

각 이미지는 하나의 레이블에 매핑되어 있습니다. 데이터셋에 클래스 이름이 들어있지 않기 때문에 나중에 이미지를 출력할 때 사용하기 위해 별도의 변수를 만들어 저장합니다.

~~~python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
~~~

---

## 데이터 탐색하기

모델을 훈련하기 전에 데이터의 구조를 살펴보도록 하겠습니다.

~~~python
print(train_images.shape)

print(len(train_labels))

print(train_labels)

print(test_images.shape)

print(len(test_labels))
~~~
###<center>결과</center>

![데이터 shape](https://drive.google.com/uc?id=1ZcIJrB9jNSx3ug_Wqqaq4p6nu7ZvZHLj)

결과창을 보시면 훈련 세트에 28 x 28 픽셀로 표현된 6만개의 이미지가 있다는 것을 알 수 있습니다

훈련 세트에 6만개의 레이블이 각각 있고, 각 레이블이 10가지의 옷의 종류를 0과 9 사이의 정수로 되어있다는 것도 알 수 있습니다.

테스트 세트에는 만개의 이미지와 그 이미지에 대한 레이블이 있다는 것을 알 수 있습니다.

---

## 데이터 전처리

네트워크를 훈련하기 전에 `데이터 전처리과정`을 거쳐야 합니다. 이 과정을 통해 분석 결과/인사이트와 모델 성능에 직접적인 영향을 미치는 과정이기 때문에 중요하게 다루어지는 과정입니다.

실무에 사용되는 데이터셋은 바로 분석이 불가능할 정도로 지저분하기 때문에 분석이 가능한 상태로 만들기 위해 전처리 방식이 자주 사용됩니다.

~~~python
plt.figure()
for i in [0, 30000, 59999]:
    plt.imshow(train_images[i])
    plt.colorbar()
    plt.grid(False)
    plt.show()
~~~
###<center>결과</center>

![0번 이미지](https://drive.google.com/uc?id=13BYqWEaihFC19u4ga6J_3X-ZlHFmMIc-)
![중간 이미지](https://drive.google.com/uc?id=1oiaRK2Us52w16BcGU3t7-WvtLAWe4zH7)
![마지막 이미지](https://drive.google.com/uc?id=1rz5TWQ5FmFr1UtLTi1iU4uv4bCvws0q1)

훈련 세트의 첫번째, 마지막, 중간 이미지를 보면 픽셀 값의 범위가 0 ~ 255 사이라는 것을 알 수 있습니다.

신경 모델에 주입하기 전에 이 값의 범위를 0 ~ 1 사이로 조정하겠습니다. 훈련 세트와 테스트 세트를 둘 다 255로 나누어 주면 됩니다.

~~~python
train_images = train_images / 255.0

test_images = test_images / 255.0
~~~

훈련 세트에서 처음 25개 이미지와 그 아래 클래스 이름을 출력해 보겠습니다. 데이터 포맷이 올바른지 확인하고 네트워크 구성과 훈련할 준비를 마칩니다.

~~~python
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
~~~
###<center>결과</center>

![25개의 이미지](https://drive.google.com/uc?id=1knMwUXYGf_62I0pPBB2rgNAjLwXHbQxa)

---

## 모델 구성

신경망 모델을 만들기 위해 모델의 층을 구성한 다음 모델을 컴파일합니다.

### 층 설정
신경망의 기본 구성 요소는 층(layer)입니다. 층은 주입된 데이터에서 표현을 추출합니다.

대부분 딥러닝은 간단한 층을 연결하여 구성됩니다. tf.keras.layers.Dense와 같은 층들의 가중치(parameter)는 훈련하는 동안 학습됩니다.

~~~python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
~~~

이 네트워크의 첫 번째 층인 tf.keras.layers.Flatten은 2차원 배열(28 x 28 픽셀)의 이미지 포맷을 28 * 28 = 784 픽셀의 1차원 배열로 변환합니다. 이 층은 이미지에 있는 픽셀의 행을 펼쳐서 일렬로 늘립니다. 이 층에는 학습되는 가중치가 없고 데이터를 변환하기만 합니다.

픽셀을 펼친 후에는 두 개의 tf.keras.layers.Dense 층이 연속되어 연결됩니다. 이 층을 밀집 연결(densely-connected) 또는 완전 연결(fully-connected) 층이라고 부릅니다. 첫 번째 Dense 층은 128개의 노드(또는 뉴런)를 가집니다. 마지막 층은 10개의 노드의 소프트맥스(softmax) 층입니다. 이 층은 10개의 확률을 반환하고 반환된 값의 전체 합은 1입니다. 각 노드는 현재 이미지가 10개 클래스 중 하나에 속할 확률을 출력합니다.

간단하게 말해서 첫째 층에서 28 x 28 픽셀을 1차원 배열로 변환하여 컴퓨터가 학습할 수 있는 형태로 변환하고
마지막 층에서 10개의 의류들을 비교하여 가장 높은 의류를 선정하는 것입니다.

---

### 모델 컴파일

모델을 훈련하기 전에 몇가지 설정을 추가합니다.
~~~python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
~~~

- 손실 함수(Loss function) : 훈련 하는 동안 모델의 오차를 측정합니다. 모델의 학습이 올바른 방향으로 향하도록 이 함수를 최소화해야 합니다.

- 옵티마이저(Optimizer) : 데이터와 손실 함수를 바탕으로 모델의 업데이트 방법을 결정합니다.

- 지표(Metrics) : 훈련 단계와 테스트 단계를 모니터링하기 위해 사용합니다. 올바르게 분류된 이미지의 비율인 정확도를 사용합니다.

### 모델 훈련

신경망 모델을 훈련하는 단계는 다음과 같습니다

1. 훈련 데이터를 모델에 주입합니다. 이 예에서는 train_images와 train_labels 배열입니다.

2. 모델이 이미지와 레이블을 매핑하는 방법을 배웁니다.

3. 테스트 세트에 대한 모델의 예측을 만듭니다. 이 예에서는 test_images 배열입니다. 이 예측이 test_labels 배열의 레이블과 맞는지 확인합니다.

model.fit 메서드를 호출하면 모델이 훈련 데이터를 학습합니다

~~~python
model.fit(train_images, train_labels, epochs=5)
~~~

### <center>결과</center>

![learning 결과](https://drive.google.com/uc?id=1scP_NsKdH50eHW-WzgI9G5Cd_9kOsOEU)

컴파일을 해보시면 모델이 훈련되면서 손실과 정확도 지표가 출력되는 것을 알 수 있습니다.

### 정확도 평가

테스트 세트에서 모델 성능을 비교합니다.

~~~python
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('테스트 정확도 : ', test_acc)
~~~

### <center>결과</center>

![learing test 결과](https://drive.google.com/uc?id=1337lLd7HlMeWmyZmLr9C6VONcKWhCfQY)

테스트 세트의 정확도가 훈련 세트의 정확도보다 조금 낮습니다. 훈련 세트의 정확도와 테스트 세트의 정확도 사이의 차이는 과대적합(overfitting) 때문입니다. 과대적합은 머신러닝 모델이 훈련 데이터보다 새로운 데이터에서 성능이 낮아지는 현상을 말합니다.

### 이미지 예측

이제 훈련된 모델을 사용하여 이미지를 예측할 수 있습니다.

~~~python
predictions = model.predict(test_images)
~~~
테스트 세트에 있는 각 이미지 레이블을 예측했습니다. 첫번째 예측결과를 살펴보면

~~~python
print(predictions[0]) # 첫 번째 이미지의 예측
np.argmax(predictions[0]) # 모델의 예측 레이블 확인
~~~

### <center>결과</center>

![첫번째 예측 확인](https://drive.google.com/uc?id=1FMuiA9BvCnjBwaKIICNZXDemaCm1jVnK)

이 예측은 10개의 숫자 배열로 나타납니다. 이 값은 10개의 옷 품목에 상응하는 모델의 신뢰도(confidence)를 나타냅니다.

모델은 이 이미지가 앵클 부츠(class_name[9])라고 가장 확신하고 있습니다. 이 값이 맞는지 테스트 레이블을 확인해 보겠습니다.

~~~python
print(test_labels[0])
~~~
### <center>결과</center>

![첫번째 예측 결과](https://drive.google.com/uc?id=1qJifGIKTjqRhj7Q39RM1PboZBi7KCXpP)

맞다는 것을 확인 할 수 있습니다.

---

이번에는 10개의 신뢰도를 그래프로 표현해보겠습니다.

~~~python
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
~~~

0번째 원소의 이미지, 예측, 신뢰도 점수 배열을 확인해 보겠습니다.

~~~python
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()
~~~

### <center>결과</center>

![첫번째 그래프](https://drive.google.com/uc?id=1AqXerxsQBRBfVTtz1utokbKqKIKsjcpM)

---

여러개의 이미지의 예측을 출력해 보겠습니다. 올바르게 예측된 레이블은 파란색이고 잘못 예측된 레이블은 빨강색입니다. 숫자는 예측 레이블의 신뢰도 퍼센트(100점 만점)입니다. 신뢰도 점수가 높을 때도 잘못 예측할 수 있습니다.

~~~python
# 처음 X 개의 테스트 이미지와 예측 레이블, 진짜 레이블을 출력합니다
# 올바른 예측은 파랑색으로 잘못된 예측은 빨강색으로 나타냅니다
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()
~~~

### <center>결과</center>

![여러개 그래프 결과](https://drive.google.com/uc?id=1Tefp_ZeggERc61ofwNS08fssaN_NHvcw)

---

마지막으로 훈련된 모델을 사용하여 한 이미지에 대한 예측을 만듭니다.

~~~python
# 테스트 세트에서 이미지 하나를 선택합니다
img = test_images[0]

print(img.shape)
~~~

### <center>결과</center>
![이미지 선택 튜플](https://drive.google.com/uc?id=1C93GliEO5hLIf49WIR6c7J9iQMnX1uZb)

`tf.keras` 모델은 한 번에 샘플의 묶음 또는 배치(batch)로 예측을 만드는데 최적화되어 있습니다. 하나의 이미지를 사용할 때에도 2차원 배열로 만들어야 합니다.

~~~python
# 이미지 하나만 사용할 때도 배치에 추가합니다
img = (np.expand_dims(img,0))

print(img.shape)
~~~

### <center>결과</center>
![2차원 배열](https://drive.google.com/uc?id=1QvjkJWXpaIs2oZyvrFv3iaBRqx9HJml7)

이제 이 이미지의 예측을 만듭니다.
~~~python
predictions_single = model.predict(img)

print(predictions_single)
~~~

### <center>결과</center>
![예측](https://drive.google.com/uc?id=1faM5kjD2niw8XzmEdvQBrNrPiPHNDtXA)

~~~python
plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()
~~~

### <center>결과</center>
![결과 그래프](https://drive.google.com/uc?id=15Ic3cHzTNOWUizRb01cAwSMitTTefckk)

`model.predict`는 2차원 넘파이 배열을 반환하므로 첫 번째 이미지의 예측을 선택합니다.

~~~python
prediction_result = np.argmax(predictions_single[0])
print(prediction_result)
~~~

### <center>결과</center>
![결과](https://drive.google.com/uc?id=1WeDV8vCVz8lTB-Ihxi_K_hCG64ctjB5I)

모델의 예측은 레이블 9, Ankle boot 입니다.
