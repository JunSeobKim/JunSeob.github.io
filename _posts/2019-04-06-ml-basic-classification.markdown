---
layout: post
title:  "머신러닝 예제 : 기초적인 분류 문제"
subtitle:   "머신러닝 기초"
categories: devlog
tags: machine_learning basic_classification
---
---
이번 포스팅에서는 코딩기초학습의 Hello World가 있는 것처럼 머신러닝의 Hello World를 예제를 통해 다루겠습니다.

이 예제는 운동화나 셔츠, 바지 같은 옷 이미지를 분류하는 신경망 모델을 훈련합니다.

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
결과

![결과](https://drive.google.com/uc?id=1EXsUjGnJ-2CdqIxp67CaV5pX38YFl5Mq "folat:left")

우선 이번 예제에서 쓸 라이브러리들을 임포트 해줍니다.
`matplotlib`는 파이썬에서 자료를 ``시각화(visulaization)``하는 패키지입니다.
특히 플롯(그래프)을 그릴 때 주로 많이 쓰이는 2D, 3D 플롯팅 패키지입니다.
