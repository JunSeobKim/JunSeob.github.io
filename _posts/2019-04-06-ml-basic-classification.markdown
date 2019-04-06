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

예제본문 <https://www.tensorflow.org/tutorials/keras/basic_classification>

---

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
<center>결과</center>

![결과](https://drive.google.com/uc?id=1EXsUjGnJ-2CdqIxp67CaV5pX38YFl5Mq)

우선 이번 예제에서 쓸 라이브러리들을 임포트 해줍니다.
`matplotlib`는 파이썬에서 자료를 ``시각화(visulaization)``하는 패키지입니다.
특히 플롯(그래프)을 그릴 때 주로 많이 쓰이는 2D, 3D 플롯팅 패키지입니다.

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

## 데이터 탐색하기

모델을 훈련하기 전에 데이터의 구조를 살펴보도록 하겠습니다.

~~~python
print(train_images.shape)

print(len(train_labels))

print(train_labels)

print(test_images.shape)

print(len(test_labels))
~~~
<center>결과</center>

![데이터 shape](https://drive.google.com/uc?id=1ZcIJrB9jNSx3ug_Wqqaq4p6nu7ZvZHLj)

결과창을 보시면 훈련 세트에 28 x 28 픽셀로 표현된 6만개의 이미지가 있다는 것을 알 수 있습니다

훈련 세트에 6만개의 레이블이 각각 있고, 각 레이블이 10가지의 옷의 종류를 0과 9 사이의 정수로 되어있다는 것도 알 수 있습니다.

테스트 세트에는 만개의 이미지와 그 이미지에 대한 레이블이 있다는 것을 알 수 있습니다.

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
<center>결과</center>

![0번 이미지](https://drive.google.com/uc?id=13BYqWEaihFC19u4ga6J_3X-ZlHFmMIc-)

훈련 세트의 첫번째, 마지막, 중간 이미지를 보면 픽셀 값의 범위가 0 ~ 255 사이라는 것을 알 수 있습니다.
