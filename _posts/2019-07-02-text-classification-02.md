---
layout: post
comments: true
title: '[NLP] 텍스트 분류 (TF-IDF) - #2'
categories: [bigdata]
tags: [ml]
date: 2019-07-02
---

이전 글에서는 데이터를 모델에 적용하기 전에 데이터에 대해 이해하고 정제하는 과정인 데이터 전처리 과정을 진행했다. 

이제 전처리된 데이터를 가지고 어랴와 같은 다양한 모델에 적용해 보자.

> **선형 회귀 모델** : 종속변수와 독립변수 간의 상관관계를 모델링하는 방법<br/>
> **로지스틱 회귀 모델** : 선형 모델의 결과값에 로지스틱 함수를 적용하여 0 ~ 1 사이의 값을 갖게 하여 확률로 표현<br/>
> **TF-IDF** : TF(Term Frequency, 단어의 빈도), IDF(역문서 빈도, Inverse Document Frequency) 쉽게 말에 문장에서 단의의 빈도수를 계산하되 너무 자주 등장하는 단어는 크게 의미를 두지 않도록 가중치를 낮게 주자는 의미.

먼저 TF-IDF를 활용해 문장 벡터를 만들 것이다. TfidfVectorizer를 사용하기 위해서는 입력값이 텍스트로 이루어진 데이터 형태여야 하기 때문에 전처리한 결과 중 numpy배열이 아닌 정제된 텍스트 데이터를 사용하자.

>훈련 데이터 : train_clean.csv<br/>
>테스트 데이터 : test_clean.csv


```python
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
```


```python
DATA_IN_PATH = './data_in/'
DATA_OUT_PATH = './data_out/'
TRAIN_CLEAN_DATA = 'train_clean.csv'
```


```python
train_data = pd.read_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA)
```


```python
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>stuff going moment mj started listening music ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>classic war worlds timothy hines entertaining ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>film starts manager nicholas bell giving welco...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>must assumed praised film greatest filmed oper...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>superbly trashy wondrously unpretentious explo...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
reviews = list(train_data['review'])
sentiments = list(train_data['sentiment'])

vectorizer = TfidfVectorizer(
            min_df=0.0,
            analyzer="char",
            sublinear_tf=True,
            ngram_range=(1,3),
            max_features=5000
        )

X = vectorizer.fit_transform(reviews)
```

- min_df : 설정한 값보다 특정 토큰의 df값이 더 적게 나오면 벡터화 과정에서 제거
- analyzer : 분석하기 위한 기준 단위(word:단어 하나를 단위로, char:문자 하나를 단위로)
- sublinear_tf : 문서의 단어 빈도 수에 대한 스무딩(smoothing) 여부
- ngram_range : 빈도의 기본 단위를 설정할 n-gram 범위
- max_features : 각 벡터의 최대 길이


```python
X
```




    <25000x5000 sparse matrix of type '<class 'numpy.float64'>'
    	with 17862871 stored elements in Compressed Sparse Row format>



## 학습과 검증 데이터셋 분리
해당 입력값을 모델에 적용하기전 학습데이터의 일부를 검증 데이터로 따로 분리하자.


```python
from sklearn.model_selection import train_test_split
import numpy as np

RANDOM_SEED = 42
TEST_SPLIT = 0.2

y = np.array(sentiments);

X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=TEST_SPLIT, 
                                                    random_state=RANDOM_SEED)
```

입력값인 X와 정답 라벨을 numpy 배열로 만든 y에 대해 적용해서 학습데이터와 검증데이터로 나누었다.

## 모델 선언 및 학습

선형 회귀 모델을 만들기 위해 LogisticRegression을 사용하고, class_weight를 'balanced'로 설정해서 각 라벨에 대해 균형 있게 학습할 수 있게 하자.


```python
from sklearn.linear_model import LogisticRegression

lgs = LogisticRegression(class_weight = 'balanced')
lgs.fit(X_train, y_train)
```

    C:\Users\nicey\.conda\envs\nlp\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    




    LogisticRegression(C=1.0, class_weight='balanced', dual=False,
                       fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                       max_iter=100, multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False)



## 검증 데이터로 성능 평가


```python
print("Accuracy : %f" % lgs.score(X_eval, y_eval)) # 검증 데이터로 성능 측정
```

    Accuracy : 0.859600
    

성능 평가 방법으로 정밀도(precision), 재현율(recall), f1-score, auc 등의 다양한 지표가 있지만 여기서는 정확도(Accuracy)만 측정하였다.

평가 결과 약 86%의 정확도를 보였다. 성능이 생각보다 나오지 않을 때는 하이퍼파라미터를 수정하거나 다른 기법들을 추가해서 성능을 올려보자. 검증 데이터의 성능이 만족할 만큼 나온다면 평가 데이터를 적용하면 된다.

## 데이터 제출하기

생성한 모델을 활용해 평가 데이터 결과를 예측하고 캐글에 제출할 수 있도록 파일로 저장하자.

우선 전처리한 텍스트 형태의 평가 데이터를 불러오자.


```python
TEST_CLEAN_DATA = 'test_clean.csv'

test_data = pd.read_csv(DATA_IN_PATH + TEST_CLEAN_DATA, header=0, delimiter=",")
```


```python
test_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>naturally film main themes mortality nostalgia...</td>
      <td>"12311_10"</td>
    </tr>
    <tr>
      <th>1</th>
      <td>movie disaster within disaster film full great...</td>
      <td>"8348_2"</td>
    </tr>
    <tr>
      <th>2</th>
      <td>movie kids saw tonight child loved one point k...</td>
      <td>"5828_4"</td>
    </tr>
    <tr>
      <th>3</th>
      <td>afraid dark left impression several different ...</td>
      <td>"7186_2"</td>
    </tr>
    <tr>
      <th>4</th>
      <td>accurate depiction small time mob life filmed ...</td>
      <td>"12128_7"</td>
    </tr>
  </tbody>
</table>
</div>



해당 데이터를 대상으로 이전에 학습 데이터에 대해 사용했던 객체를 사용해 TF-IDF 값으로 벡터화한다.


```python
testDataVecs = vectorizer.transform(test_data['review'])
```

백터화할 때 평가 데이터에 대해서는 fit을 호출하지 않고 그대로 transform만 호툴한다.

이미 학습 데이터에 맞게 설정했고, 그 설정에 맞게 평가 데이터도 변환을 하면 된다.

이제 이 값으로 예측한 후 예측값을 하나의 변수로 할당하고 출력해보자.


```python
test_predicted = lgs.predict(testDataVecs)
print(test_predicted)
```

    [1 0 1 ... 0 1 0]
    

결과를 보면 각 데이터에 대해 긍정, 부정 값을 가지고 있다.

이제 이 값을 캐글에 제출하기 위해 csv 파일로 저장하자. 캐글에 제출하기 위한 데이터 형식은 각 데이터의 고유한 id 값과 결과값으로 구성되어야 한다.


```python
DATA_OUT_PATH = './data_out/'

if not os.path.exists(DATA_OUT_PATH):
    os.makedirs(DATA_OUT_PATH)
    
ids = list(test_data['id'])

answer_dataset = pd.DataFrame({'id': test_data['id'], 'sentiment': test_predicted})
answer_dataset.to_csv(DATA_OUT_PATH + 'lgs_tfidf_answer.csv', index=False, quoting=3)
```


