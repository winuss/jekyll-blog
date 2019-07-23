---
layout: post
comments: true
title: '[TF] 텐서플로우 소개 및 맛보기'
categories: [bigdata]
tags: [ml]
date: 2019-07-21
---

텐서플로우(TensorFlow)는 데이터 플로우 그래프(Data flow graph)를 사용하여 수치 연산을 하는 오픈소스 소프트웨어 라이브러리입니다.


### 데이터 플로우 그래프(Data flow graph)

![mnist](/assets/img/post/tensorflow-basic/data-flow-graph.gif)

수학 계산과 데이터의 흐름을 노드(Node)와 엣지(Edge)를 사용한 양방향 그래프(Directed Graph)로 표현합니다.

- 노드(Node) : 수치 연산
- 엣지(Edge) : 노드 사이를 이동하는 다차원 데이터 배역(텐서, tensor)

### 특징 및 장점

텐서플로우는 머신러닝과 딥 뉴럴 네트워크 연구를 목적으로 구글의 인공지능 연구 조직인 구글 브레인 팀의 연구자와 엔지니어들에 의해 개발었는데 다임과 같은 특징을 가집니다.

- 코드 수정없이 데스크탑, 서버 혹은 모바일 디바이스에서 CPU나 GPU를 사용하여 연산을 구동
- 분산(distributed) 실행환경이 가능
- 아이디어 테스트에서 서비스 단계까지 모두 이용가능
- 계산 구조와 목표 함수만 정의하면 자동으로 미분 계산 처리



### 설치

먼저 아나콘다를 설치한 후 다음과 같이 conda install 명령으로 간단하게 설치할 수 있습니다.

`conda install tensorflow`

## 실습 

그럼 이제 텐서플로우 공식 홈페이지에 나와있는 튜토리얼을 따라 해보도록 하겠습니다.

### Basic classification

<https://www.tensorflow.org/tutorials/keras/basic_classification?hl=ko>

### 데이터 준비하기


```python
import tensorflow as tf
from tensorflow import keras
```


```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
print(tf.__version__) 
```

    1.13.1
    

현재 사용하는 버전은 1.13.1 입니다.

이제 10개의 범주(category)와 70,000개의 흑백 이미지로 구성된 패션 MNIST 데이터셋을 사용하겠습니다. 이미지는 해상도(28x28 픽셀)가 낮고 다음처럼 개별 옷 품목을 나타냅니다.

![mnist](/assets/img/post/tensorflow-basic/fashion-mnist-sprite.png){: width="600"}


```python
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
    32768/29515 [=================================] - 0s 1us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
    26427392/26421880 [==============================] - 2s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
    8192/5148 [===============================================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
    4423680/4422102 [==============================] - 1s 0us/step
    

Fashion MNIST data를 가져와 훈련셋(train_images, train_labels)와 테스트셋(test_images, test_labels)에 저징합니다.

우리가 만드는 학습모델에 훈련셋을 통해 학습시킬 것이고, 테스트셋의 test_images을 통해 예측값을 도출한 후 test_labels와 비교하여 오차를 측정할 것입니다.


```python
train_images.shape
```




    (60000, 28, 28)




```python
test_images.shape
```




    (10000, 28, 28)



데이터를 간단히 살펴보면 훈련셋에는 60000개, 테스트셋에는 1000개의 이미지 이고 하나의 이미지는 28x28 크기를 가지고 있는것을 확인 할 수 있습니다.


```python
train_labels
```




    array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)



위와 같이 label은 0~9까지의 값을 가지고 있다. 각 숫자가 의미하는 바는 다음과 같습니다.

**Label: Class**

0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot


```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

그럼 불러온 fashion_mnist 데이터를 살펴본 결과를 그림으로 확인해 보죠.

![mnist](/assets/img/post/tensorflow-basic/load-data.png)

### 데이터 전처리

모델에 데이터를 학습시키기 위해 먼저 데이터 전처리를 진행해 보겠습니다.


```python
plt.figure()
# train_images의 첫번째 요소를 그림
plt.imshow(train_images[0])
plt.colorbar()
# 점자선을 없애자.
plt.grid(False)
plt.show()
```


![png](/assets/img/post/tensorflow-basic/output_18_0.png)


픽셀 값의 범위는 0 ~ 255 입니다.

신경망 모델에 주입하기 전에 이 값의 범위를 0~1 사이로 조정해야 하기 때문에 모든 값을 255로 나눠 줍니다.
(색상은 무시 하겠다는, 즉 Gray scale로 하겠다는 의미)


```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

위와 같은 그림을 다시한번 확인해 보면 모든 요소가 0~1 사이의 값을 갖도록 변한 것을 확인할 수 있습니다.


```python
# matplotlib을 통해 그림을 그린다.
plt.figure()
# train_images의 첫번째 요소를 그린다.
plt.imshow(train_images[0])
plt.colorbar()
# 점자선을 False로 둠으로써 없앤다.
plt.gca().grid(False)
```


![png](/assets/img/post/tensorflow-basic/output_23_0.png)


보다 많은 이미지를 확인해 보기 위해 위에서 정의했던 `class_names`를 적용해 어떤 사진인지 label과 함께 확인해 봅니다.


```python
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```


![png](/assets/img/post/tensorflow-basic/output_25_0.png)


### 모델 구성

이제 실제로 Fashion MNIST data를 예측하는 모델을 만들어 보겠습니다.

신경망 모델을 만들려면 모델의 층을 구성한 다음 모델을 컴파일 해야 합니다.

**층 설정**

신경망의 기본 구성요소는 층(layer)입니다. 층은 주입된 데이터에서 표현을 추출합니다. 아마도 문제를 해결하는데 더 의미있는 표현이 추출될 것입니다.

대부분 딥러닝은 간단한 층을 연결하여 구성됩니다. **tf.keras.layers.Dense**와 같은 층들의 가중치(parameter)는 훈련하는 동안 학습됩니다.


```python
model = keras.Sequential([
    #2차원배열(28x28 픽셀)의 이미지 포맷을 28*28=784 픽셀의 1차원 배열로 변환
    keras.layers.Flatten(input_shape=(28,28)),
    #128개의 노드(뉴런)
    keras.layers.Dense(128, activation=tf.nn.relu),
    #10개의 노드의 소프트맥스층 (10개의 확률을 반환하고 반환된 값은 전체 합의 1이됨)
    #각 노드는 현재 이미지가 10개 클래스 중 하나에 속할 확률을 출력
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

    WARNING:tensorflow:From C:\Users\nicey\.conda\envs\nlp\lib\site-packages\tensorflow\python\ops\resource_variable_ops.py:435: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    

위의 코드를 확인해보면 model은 총 3개의 layer를 갖는 것을 확인할 수 있는데,

첫번째 layer, Flatten 에서는 28x28로 되어있는 2차원 값을 1차원으로 변환.

두번째 layer, Dense 에서는 128개의 노드를 가지며 relu라는 activation function을 수행.

세번째 lyaer, Dense 에서는 10개의 노드를 가지며 softmax함수를 통해 classification하는 작업을 수행.

### 모델 컴파일

모델을 훈련하기 전에 필요한 몇 가지 설정이 모델 컴파일 단계에서 추가됩니다:

- 손실 함수(Loss function)-훈련 하는 동안 모델의 오차를 측정합니다. 모델의 학습이 올바른 방향으로 향하도록 이 함수를 최소화해야 합니다.
- 옵티마이저(Optimizer)-데이터와 손실 함수를 바탕으로 모델의 업데이트 방법을 결정합니다.
- 지표(Metrics)-훈련 단계와 테스트 단계를 모니터링하기 위해 사용합니다. 다음 예에서는 올바르게 분류된 이미지의 비율인 정확도를 사용합니다.


```python
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

모델이 사용하는 optimizer를 선택해주고, loss function을 선택, metrics는 측정항목을 적어 줍니다.

### 모델 훈련

신경망 모델을 훈련하는 단계는 다음과 같습니다:

1. 훈련 데이터를 모델에 주입합니다-이 예에서는 train_images와 train_labels 배열입니다.
2. 모델이 이미지와 레이블을 매핑하는 방법을 배웁니다.
3. 테스트 세트에 대한 모델의 예측을 만듭니다-이 예에서는 test_images 배열입니다. 이 예측이 test_labels 배열의 레이블과 맞는지 확인합니다.

훈련을 시작하기 위해 model.fit 메서드를 호출하면 모델이 훈련 데이터를 학습합니다:

fit 함수를 이용해 학습시킬 images와 labels를 넣어주고 epoch를 설정합니다.


```python
model.fit(train_images, train_labels, epochs=5)
```

    Epoch 1/5
    60000/60000 [==============================] - 3s 58us/sample - loss: 0.5000 - acc: 0.8243
    Epoch 2/5
    60000/60000 [==============================] - 3s 53us/sample - loss: 0.3762 - acc: 0.8633
    Epoch 3/5
    60000/60000 [==============================] - 3s 51us/sample - loss: 0.3366 - acc: 0.8770
    Epoch 4/5
    60000/60000 [==============================] - 3s 53us/sample - loss: 0.3162 - acc: 0.8841
    Epoch 5/5
    60000/60000 [==============================] - 3s 52us/sample - loss: 0.2963 - acc: 0.8911
    




    <tensorflow.python.keras.callbacks.History at 0x237f2119e10>



위 코드를 실행하면 학습되는 상태정보를 순차적으로 확인할 수 있습니다.

### 정확도 평가

이제 model의 evaluate함수를 이용하여 test images와 labels를 인자로 주어 테스트 세트에서 모델의 성능을 비교합니다.


```python
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('테스트 정확도:', test_acc)
```

    10000/10000 [==============================] - 0s 28us/sample - loss: 0.3484 - acc: 0.8758
    테스트 정확도: 0.8758
    

테스트 세트의 정확도가 훈련 세트의 정확도보다 조금 낮습니다. 훈련 세트의 정확도와 테스트 세트의 정확도 사이의 차이는 과대적합(overfitting) 때문입니다. 과대적합은 머신러닝 모델이 훈련 데이터보다 새로운 데이터에서 성능이 낮아지는 현상을 말합니다.

### 예측 만들기

훈련된 모델을 사용하여 실제 모델이 어떻게 예측하는지 test_images로 확인할 수 있습니다.


```python
predictions = model.predict(test_images)
```

테스트 세트에 있는 각 이미지의 레이블을 예측했습니다. 첫 번째 예측을 확인해 봅니다.


```python
predictions[0]
```




    array([1.1595419e-08, 1.3519094e-10, 2.1264961e-09, 1.2290319e-10,
           7.7952078e-10, 4.4512744e-03, 2.5239646e-07, 7.9262927e-03,
           1.2780076e-07, 9.8762208e-01], dtype=float32)



이 예측은 10개의 숫자 배열로 나타납니다. 이 값은 10개의 옷 품목에 상응하는 모델의 신뢰도(confidence)를 나타냅니다. 가장 높은 신뢰도를 가진 레이블을 찾습니다.

우리는 해당 10개의 값중 가장 큰 값을 정답이라고 생각할 것이기 때문에 numpy의 argmax함수를 이용해 가장 큰 값을 갖는 인덱스를 찾습니다.


```python
np.argmax(predictions[0])
```




    9



모델은 이 이미지가 앵클 부츠(class_name[9])라고 가장 확신하고 있습니다. 이 값이 맞는지 테스트 레이블을 확인해 보죠:


```python
test_labels[0]
```




    9



오! 정답..^^

위와 같은 방식으로 총 25개의 test_images에 대한 이미지와 예측결과를 비교해 보겠습니다.


```python
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    # predictions에서 가장 큰 값을 predicted_label 로 가져온다.
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    # 이때 실제 test_label과 일치하면 초록색 글씨로,
    if predicted_label == true_label:
      color = 'green'
    # 일치하지 않으면 빨간색 글씨로 출력한다.
    else:
      color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label], 
                                  class_names[true_label]),
                                  color=color)
```


![png](/assets/img/post/tensorflow-basic/output_49_0.png)


빨간색 글씨가 3개로 확인됩니다. 오답이 3개 나왔네요.

이제 10개의 신뢰도를 모두 그래프로 표현해 보겠습니다.


```python
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
```

0번째 원소의 이미지, 예측, 신뢰도 점수 배열을 확인해 보겠습니다.


```python
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()
```


![png](/assets/img/post/tensorflow-basic/output_54_0.png)



```python
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()
```


![png](/assets/img/post/tensorflow-basic/output_55_0.png)


몇 개의 이미지의 예측을 출력해 보죠. 올바르게 예측된 레이블은 파란색이고 잘못 예측된 레이블은 빨강색입니다. 숫자는 예측 레이블의 신뢰도 퍼센트(100점 만점)입니다. 신뢰도 점수가 높을 때도 잘못 예측할 수 있습니다.


```python
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
```


![png](/assets/img/post/tensorflow-basic/output_57_0.png)


마지막으로 훈련된 모델을 사용하여 한 이미지에 대한 예측을 만듭니다.


```python
# 테스트 세트에서 이미지 하나를 선택합니다
img = test_images[0]

print(img.shape)
```

    (28, 28)
    

tf.keras 모델은 한 번에 샘플의 묶음 또는 배치(batch)로 예측을 만드는데 최적화되어 있습니다. 하나의 이미지를 사용할 때에도 2차원 배열로 만들어야 합니다:


```python
# 이미지 하나만 사용할 때도 배치에 추가합니다
img = (np.expand_dims(img,0))

print(img.shape)
```

    (1, 28, 28)
    

이제 이 이미지의 예측을 만듭니다:


```python
predictions_single = model.predict(img)

print(predictions_single)
```

    [[2.7692769e-05 5.9337044e-08 3.9947031e-06 2.0716065e-08 5.4740326e-06
      2.3610677e-01 1.6528367e-05 2.1145634e-01 4.4389912e-05 5.5233872e-01]]
    


```python
plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()
```


![png](/assets/img/post/tensorflow-basic/output_64_0.png)


model.predict는 2차원 넘파이 배열을 반환하므로 첫 번째 이미지의 예측을 선택합니다:


```python
prediction_result = np.argmax(predictions_single[0])
print(prediction_result)
```

    9
    

이전과 마찬가지로 모델의 예측은 레이블 9입니다.
