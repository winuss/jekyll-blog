---
layout: post
comments: true
title: '[TF] 텍스트 분류'
categories: [bigdata]
tags: [ml]
date: 2019-07-21
---

지난번에 TF-IDF 및 Word2Vec을 이용해 텍스트를 분류해 보았다. 이번에는 텐서플로를 이용하여 텍스트를 분류 해보자.

여기에서는 인터넷 영화 데이터베이스(Internet Movie Database)에서 수집한 50,000개의 영화 리뷰 텍스트를 담은 IMDB 데이터셋을 사용하겠다. 25,000개 리뷰는 훈련용으로, 25,000개는 테스트용으로 나뉘어져 있고, 훈련 세트와 테스트 세트의 클래스는 균형이 잡혀 있다. 즉 긍정적인 리뷰와 부정적인 리뷰의 개수가 동일하다.

데이터는 Keras에서 제공하는 datasets를 이용하여 다운로드 하자.

## Text Classification

https://www.tensorflow.org/tutorials/keras/basic_text_classification?hl=ko


```python
from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals
import tensorflow as tf
from tensorflow import keras
```


```python
import numpy as np
print(tf.__version__)
print(np.__version__)
```

    1.13.1
    1.16.2
    

만약 numpy 버전이 1.16.2 이상의 버전이라면 삭제 후 버전을 지정하여 설치를 해야 한다. 

(imdb.load_data를 하는 도중 오류 발생)

> conda install numpy=1.16.2

### 데이터 준비

- keras dataset에 있는 imdb를 사용한다.
- train과 test 데이터 셋은 각각 25,000개 이며 데이터는 review 자체로 구성
- labels는 0또는 1값으로 긍정 부정을 나타냄

> test데이터를 이용하여 해당 Review가 영화에 대해 긍정 또는 부정적인지를 예측해 본다.


```python
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
```


```python
print("훈련 샘플: {}, 레이블: {}".format(len(train_data), len(train_labels)))
```

    훈련 샘플: 25000, 레이블: 25000
    

실제 데이터를 살펴보면 문자가 아닌 숫자 값이 리스트로 들어있다.


```python
print(train_data[0])
```

    [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]
    

이렇게 되어 있는 이유는 추가적인 dictionary에 각 숫자와 단어가 매칭되어 있기 때문이다. 또한 아래와 같이 각각의 데이터 길이도 다른것을 확인 할 수 있다.


```python
len(train_data[0]), len(train_data[1])
```




    (218, 189)



하지만 실제로 ML모델에 넣어줄때는 입력 길이는 모두 같아야 한다. 따라서 입력길이를 모두 동일하게 해주는 작업이 필요하다. 먼저 데이터를 보다 자세히 확인해보기 위해 각 데이터의 숫자를 단어로 치환해 보자.


```python
# 숫자로 된 값을 단어로 바꾸기 위한 dictionary를 가져온다.
word_index = imdb.get_word_index()
```

- word_index
> {'fawn': 34701,
> 'tsukino': 52006,
> 'nunnery': 52007,
> 'sonja': 16816,
> 'vani': 63951,
> 'woods': 1408,
> ...
>}

- word_index.items()
> dict_items([('fawn', 34701), ('tsukino', 52006), 
('nunnery', 52007), ('sonja', 16816), 
('vani', 63951), ('woods', 1408), 
('spiders', 16115) ,,,

pad, start, unknown, unused 값을 나타내기 위해 각 value에 3을 더하고 비어있게 되는 0~3에 각각을 할당한다.


```python
# 처음 몇 개 인덱스는 사전에 정의되어 있습니다
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '<'+str(i)+'>?') for i in text])
```

실제로 필요한 dictionary는 숫자가 key이고, 단어가 value인 dictionary이기 때문에 reverse_word_index라는 dictionary를 구성하고 숫자로 이루어진 입력데이터를 단어로 치환해 주며 문장으로 출력하는 **decode_review** 함수를 만든다.

## decode example
- [1,14,22,16,43,530...] => "<START> <14>? <22>? <16>? film flick..."
- 1 : `<START>`
- 14, 22, 16 : unknown
- 43 : film
- 530 : flick


```python
decode_review(train_data[0])
```




    "<START> this film was just brilliant casting location scenery story direction everyone's really suited the part they played and you could just imagine being there robert <UNK> is an amazing actor and now the same being director <UNK> father came from the same scottish island as myself so i loved the fact there was a real connection with this film the witty remarks throughout the film were great it was just brilliant so much that i bought the film as soon as it was released for <UNK> and would recommend it to everyone to watch and the fly fishing was amazing really cried at the end it was so sad and you know what they say if you cry at a film it must have been good and this definitely was also <UNK> to the two little boy's that played the <UNK> of norman and paul they were just brilliant children are often left out of the <UNK> list i think because the stars that play them all grown up are such a big profile for the whole film but these children are amazing and should be praised for what they have done don't you think the whole story was so lovely because it was true and was someone's life after all that was shared with us all"



앞서 추가해주었던 `START`, `UNK` 등이 추가되어 보여지는 것을 확인할 수 있다.

## 데이터 전치리

위에서 언급했던 각 데이터의 길이가 상이한 것을 처리한다. keras에서 제공하는 preprocessing 함수를 이용하여 모든 데이터를 최대길이로 늘려주면서 빈공간에는 위에서 dictionary에 추가적으로 넣어주었던 pad값을 이용한다.

신경망에 주입하기 전에 텐서로 변환되어야 하는데, 변환하는 방법에는 몇 가지가 있다.

- 원-핫 인코딩(one-hot encoding) : 정수 배열을 0과 1로 이루어진 벡터로 변환한다. 예를 들어 배열 [3, 5]을 인덱스 3과 5만 1이고 나머지는 모두 0인 10,000차원 벡터로 변환할 수 있다. 그다음 실수 벡터 데이터를 다룰 수 있는 층-Dense 층-을 신경망의 첫 번째 층으로 사용한다. 이 방법은 num_words * num_reviews 크기의 행렬이 필요하기 때문에 메모리를 많이 사용하게 된다는 단점이 있다.
- 다른 방법으로는, 정수 배열의 길이가 모두 같도록 패딩(padding)을 추가해 max_length * num_reviews 크기의 정수 텐서를 만드는 방법이다. 이런 형태의 텐서를 다룰 수 있는 임베딩(embedding) 층을 신경망의 첫 번째 층으로 사용할 수 있다.

여기서는 두 번째 방식을 사용해보자.

영화 리뷰의 길이가 같아야 하므로 **pad_sequences** 함수를 사용해 길이를 맞추자.


```python
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
```

이 작업을 통해 변경된 것을 확인해 보면,

1. train_data, test_data 길이가 동일
2. 배열내 모든 데이터가 256
3. 데이터 형태는 맨 뒤에 0값, 즉 pad값이 포함되어 있음


```python
len(train_data[0]), len(train_data[1])
```




    (256, 256)




```python
print(train_data[0])
```

    [   1   14   22   16   43  530  973 1622 1385   65  458 4468   66 3941
        4  173   36  256    5   25  100   43  838  112   50  670    2    9
       35  480  284    5  150    4  172  112  167    2  336  385   39    4
      172 4536 1111   17  546   38   13  447    4  192   50   16    6  147
     2025   19   14   22    4 1920 4613  469    4   22   71   87   12   16
       43  530   38   76   15   13 1247    4   22   17  515   17   12   16
      626   18    2    5   62  386   12    8  316    8  106    5    4 2223
     5244   16  480   66 3785   33    4  130   12   16   38  619    5   25
      124   51   36  135   48   25 1415   33    6   22   12  215   28   77
       52    5   14  407   16   82    2    8    4  107  117 5952   15  256
        4    2    7 3766    5  723   36   71   43  530  476   26  400  317
       46    7    4    2 1029   13  104   88    4  381   15  297   98   32
     2071   56   26  141    6  194 7486   18    4  226   22   21  134  476
       26  480    5  144   30 5535   18   51   36   28  224   92   25  104
        4  226   65   16   38 1334   88   12   16  283    5   16 4472  113
      103   32   15   16 5345   19  178   32    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0]
    

## 모델 구성하기

이제 text classification을 수행할 ML모델을 만들어보자.

vocab_size는 영화리뷰에 사용되는 단어의 개수이다. 실제로 위에서 단어와 숫자를 매칭하는 dictionary의 사이즈 보다 크지만, 해당 데이터에서는 10000개의 단어 이내로 리뷰가 작성되었다. 

각 레이어에 대한 설명은 다음과 같다.

1. **embedding** : 숫자로 인코딩 되어있는 각 단어를 사용하며 각 단어 인덱스에 대한 벡터를 찾는다. 이러한 데이터는 추후 모델이 핫습하는데 사용된다.
2. **GlobalAveragePooling1D** : 각 예시에 대해 sequence 차원을 평균하여 고정된 길이의 벡터를 출력한다.
3. **Dense** : 첫번째 Dense 레이어를 통해서 고정길이로 출력된 vector값을 통해 16개의 hidden unit을 가진 fully-connected layer를 통과시킨다. 이후 두번째 Dense 레이어는 단일 출력 노드를 가지고 시그모이드 활성화 함수를 사용함으로써 결과에 대해 0~1 사이의 값을 가지도록 한다.


```python
# 입력 크기는 영화 리뷰 데이터셋에 적용된 어휘 사전의 크기입니다(10,000개의 단어)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16, input_shape=(None,)))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, None, 16)          160000    
    _________________________________________________________________
    global_average_pooling1d_2 ( (None, 16)                0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 16)                272       
    _________________________________________________________________
    dense_5 (Dense)              (None, 1)                 17        
    =================================================================
    Total params: 160,289
    Trainable params: 160,289
    Non-trainable params: 0
    _________________________________________________________________
    

모델 구성에 마지막으로 loss function과 optimizer를 설정한다.


```python
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['acc'])
```

## 모델 훈련하기

모델을 훈련하기에 앞서 10000개의 데이터를 분리하여 validation set을 만들자.
모델이 새롭게 접하는 데이터에 대한 accuracy와 loss등을 확인하기 위함이다.


```python
# 검증 세트 만들기
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]
```


```python
# 모델 훈련
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
```

    Train on 15000 samples, validate on 10000 samples
    Epoch 1/40
    15000/15000 [==============================] - 1s 99us/sample - loss: 0.6925 - acc: 0.5343 - val_loss: 0.6918 - val_acc: 0.5347
    Epoch 2/40
    15000/15000 [==============================] - 1s 69us/sample - loss: 0.6899 - acc: 0.6281 - val_loss: 0.6880 - val_acc: 0.6741
    Epoch 3/40
    15000/15000 [==============================] - 1s 70us/sample - loss: 0.6835 - acc: 0.7101 - val_loss: 0.6792 - val_acc: 0.7429
    Epoch 4/40
    15000/15000 [==============================] - 1s 70us/sample - loss: 0.6699 - acc: 0.7408 - val_loss: 0.6623 - val_acc: 0.7469
    Epoch 5/40
    15000/15000 [==============================] - 1s 71us/sample - loss: 0.6463 - acc: 0.7775 - val_loss: 0.6360 - val_acc: 0.7647
    Epoch 6/40
    15000/15000 [==============================] - 1s 71us/sample - loss: 0.6127 - acc: 0.7950 - val_loss: 0.6017 - val_acc: 0.7821
    Epoch 7/40
    15000/15000 [==============================] - 1s 71us/sample - loss: 0.5707 - acc: 0.8089 - val_loss: 0.5601 - val_acc: 0.7975
    Epoch 8/40
    15000/15000 [==============================] - 1s 70us/sample - loss: 0.5240 - acc: 0.8279 - val_loss: 0.5180 - val_acc: 0.8142
    Epoch 9/40
    15000/15000 [==============================] - 1s 72us/sample - loss: 0.4775 - acc: 0.8426 - val_loss: 0.4773 - val_acc: 0.8298
    Epoch 10/40
    15000/15000 [==============================] - 1s 72us/sample - loss: 0.4343 - acc: 0.8581 - val_loss: 0.4416 - val_acc: 0.8392
    Epoch 11/40
    15000/15000 [==============================] - 1s 72us/sample - loss: 0.3966 - acc: 0.8707 - val_loss: 0.4114 - val_acc: 0.8505
    Epoch 12/40
    15000/15000 [==============================] - 1s 72us/sample - loss: 0.3645 - acc: 0.8798 - val_loss: 0.3880 - val_acc: 0.8537
    Epoch 13/40
    15000/15000 [==============================] - 1s 71us/sample - loss: 0.3385 - acc: 0.8867 - val_loss: 0.3670 - val_acc: 0.8615
    Epoch 14/40
    15000/15000 [==============================] - 1s 71us/sample - loss: 0.3154 - acc: 0.8932 - val_loss: 0.3519 - val_acc: 0.8651
    Epoch 15/40
    15000/15000 [==============================] - 1s 71us/sample - loss: 0.2967 - acc: 0.8977 - val_loss: 0.3393 - val_acc: 0.8694
    Epoch 16/40
    15000/15000 [==============================] - 1s 71us/sample - loss: 0.2800 - acc: 0.9027 - val_loss: 0.3290 - val_acc: 0.8711
    Epoch 17/40
    15000/15000 [==============================] - 1s 72us/sample - loss: 0.2651 - acc: 0.9080 - val_loss: 0.3207 - val_acc: 0.8753
    Epoch 18/40
    15000/15000 [==============================] - 1s 72us/sample - loss: 0.2519 - acc: 0.9119 - val_loss: 0.3132 - val_acc: 0.8780
    Epoch 19/40
    15000/15000 [==============================] - 1s 72us/sample - loss: 0.2401 - acc: 0.9159 - val_loss: 0.3070 - val_acc: 0.8791
    Epoch 20/40
    15000/15000 [==============================] - 1s 71us/sample - loss: 0.2295 - acc: 0.9203 - val_loss: 0.3029 - val_acc: 0.8798
    Epoch 21/40
    15000/15000 [==============================] - 1s 72us/sample - loss: 0.2190 - acc: 0.9240 - val_loss: 0.2990 - val_acc: 0.8792
    Epoch 22/40
    15000/15000 [==============================] - 1s 72us/sample - loss: 0.2101 - acc: 0.9261 - val_loss: 0.2951 - val_acc: 0.8828
    Epoch 23/40
    15000/15000 [==============================] - 1s 72us/sample - loss: 0.2011 - acc: 0.9291 - val_loss: 0.2934 - val_acc: 0.8819
    Epoch 24/40
    15000/15000 [==============================] - 1s 72us/sample - loss: 0.1933 - acc: 0.9325 - val_loss: 0.2905 - val_acc: 0.8837
    Epoch 25/40
    15000/15000 [==============================] - 1s 71us/sample - loss: 0.1853 - acc: 0.9369 - val_loss: 0.2883 - val_acc: 0.8846
    Epoch 26/40
    15000/15000 [==============================] - 1s 71us/sample - loss: 0.1783 - acc: 0.9395 - val_loss: 0.2880 - val_acc: 0.8839
    Epoch 27/40
    15000/15000 [==============================] - 1s 71us/sample - loss: 0.1715 - acc: 0.9435 - val_loss: 0.2868 - val_acc: 0.8846
    Epoch 28/40
    15000/15000 [==============================] - 1s 71us/sample - loss: 0.1650 - acc: 0.9460 - val_loss: 0.2861 - val_acc: 0.8845
    Epoch 29/40
    15000/15000 [==============================] - 1s 71us/sample - loss: 0.1596 - acc: 0.9487 - val_loss: 0.2868 - val_acc: 0.8839
    Epoch 30/40
    15000/15000 [==============================] - 1s 71us/sample - loss: 0.1537 - acc: 0.9511 - val_loss: 0.2859 - val_acc: 0.8855
    Epoch 31/40
    15000/15000 [==============================] - 1s 71us/sample - loss: 0.1477 - acc: 0.9538 - val_loss: 0.2861 - val_acc: 0.8857
    Epoch 32/40
    15000/15000 [==============================] - 1s 70us/sample - loss: 0.1423 - acc: 0.9558 - val_loss: 0.2867 - val_acc: 0.8857
    Epoch 33/40
    15000/15000 [==============================] - 1s 70us/sample - loss: 0.1370 - acc: 0.9571 - val_loss: 0.2882 - val_acc: 0.8858
    Epoch 34/40
    15000/15000 [==============================] - 1s 70us/sample - loss: 0.1324 - acc: 0.9605 - val_loss: 0.2892 - val_acc: 0.8868
    Epoch 35/40
    15000/15000 [==============================] - 1s 70us/sample - loss: 0.1282 - acc: 0.9607 - val_loss: 0.2907 - val_acc: 0.8871
    Epoch 36/40
    15000/15000 [==============================] - 1s 70us/sample - loss: 0.1234 - acc: 0.9636 - val_loss: 0.2923 - val_acc: 0.8866
    Epoch 37/40
    15000/15000 [==============================] - 1s 70us/sample - loss: 0.1189 - acc: 0.9657 - val_loss: 0.2941 - val_acc: 0.8860
    Epoch 38/40
    15000/15000 [==============================] - 1s 70us/sample - loss: 0.1148 - acc: 0.9671 - val_loss: 0.2968 - val_acc: 0.8852
    Epoch 39/40
    15000/15000 [==============================] - 1s 70us/sample - loss: 0.1113 - acc: 0.9683 - val_loss: 0.2990 - val_acc: 0.8846
    Epoch 40/40
    15000/15000 [==============================] - 1s 71us/sample - loss: 0.1073 - acc: 0.9699 - val_loss: 0.3009 - val_acc: 0.8841
    

Epoch 1에서 40까지 진행이 되고 해당 Epoch에서의 acc 및 loss를 확인할 수 있다.

이제 모델 평가 결과를 확인해 보자.


```python
results = model.evaluate(test_data, test_labels)
print(results)
```

    25000/25000 [==============================] - 1s 21us/sample - loss: 0.3204 - acc: 0.8736
    [0.3203906687307358, 0.8736]
    

실제로 더 진보된 모델이라고 하기 위해서는 약 95% 이상의 정확도를 필요로 한다. 하지만 매우 단순한 방식을 사용했기 때문에 약 87% 정도의 정확도를 달성한 것을 확인 할 수 있다.

결과를 정확도와 오차의 그래프로 확인해 보자.

## 정확도와 손실 그래프 그리기

`model.fit()`은 History 객체를 반환한다. 여기에는 훈련하는 동안 일어난 모든 정보가 dictionary로 담겨 있다.


```python
history_dict = history.history
history_dict.keys()
```




    dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])



키를 확인해 보면 네개의 항목이 있다. 이는 훈련과 검증 단계에서 모니터링하는 지표들이다. 훈련 손실과 검증 손실을 그래프로 그려 보고, 훈련 정확도와 검증 정확도도 그래프로 그려 비교해 보자.


```python
import matplotlib.pyplot as plt

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# 'bo'는 파란색점, 'b'는 파란 실선
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

```


![png](/assets/img/post/tf-text-classification/output_39_0.png)



```python
plt.clf()   # 그림을 초기화합니다

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
```


![png](/assets/img/post/tf-text-classification/output_40_0.png)


점선은 훈련 loss와 accuracy이고 실선은 검증 loss와 accuracy를 나타낸다.

훈련(점선)에서 loss는 epoch가 진행되면서 감소하고 accuracy는 증가한다. 하지만 검증(실선)에서는 약 20번째 epoch 이후가 최적점인 것 같다. 이는 훈련 데이터에서만 잘 동작하는 과대적합 때문일 것이다. 이 지점 부터는 모델이 과도하게 최적화되어 테스트 데이터에서 일반화되기 어려운 훈련 데이터 특정 표현을 학습하게 된다.

여기에서는 과대적합을 막기 위해 단순히 20번째 epoch 근처에서 훈련을 멈출 수 있다.