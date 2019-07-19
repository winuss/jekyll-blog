---
layout: post
comments: true
title: '[NLP] 텍스트 분류 (Word2Vec) - #3'
categories: [bigdata]
tags: [ml]
date: 2019-07-19
---

이번에는 word2vec을 활용하여 모델을 구현 해보자.

### word2vec을 활용한 모델 구현

word2vec을 활용해 모델을 만들기 위해서는 먼저 각 단어에 대해 word2vec으로 백터화해야 한다.

word2vec의 경우 단어로 표현된 리스트를 입력값으로 넣어야 한다.


```python
import pandas as pd
```


```python
DATA_IN_PATH = './data_in/'
TRAIN_CLEAN_DATA = 'train_clean.csv'

train_data = pd.read_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA)

reviews = list(train_data['review'])
sentiments = list(train_data['sentiment'])

sentences = []
for review in reviews:
    sentences.append(review.split()) #review문장내 단어들을 배열로 저장(뜨어쓰기 기준)
```

### word2vec 백터화

이제 word2vec 모델 학습을 진행하기 앞서 word2vec 모델의 하이퍼 파라미터를 설정해야 한다.


```python
# 학습 시 필요한 하이퍼 파라미터
num_features = 300    # 워드 백터 특정값 수
min_word_count = 40   # 단어에 대한 최소 빈도 수
num_workers = 4       # 프로세스 개수
context = 10         # 컨텍스트 윈도우 크기
downsampling = 1e-3   # 다운 샘플링 비율
```

- num_fratures : 각 단어에 대한 임베딩된 벡터의 차원을 정한다.
- min_word_count : 모델에 의미 있는 단어를 가지고 학습하기 위해 적은 빈도 수의 단어들은 학습하지 않는다.
- num_workers : 모델에 의미 있는 단어를 가지고 학습하기 위해 적은 빈도 수의 단어들은 학습하지 않는다.
- context : word2vec을 수행하기 위한 컨텍스트 윈도우 크기를 지정한다.
- downsampling : word2vec 학습을 수행할 때 더 빠른 학습을 위해 정답 단어 라벨에 대한 다운샘플링 비율을 지정한다. (보통 0.001이 좋은 성능을 낸다고 한다)


```python
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
   level=logging.INFO)
```

로깅을 할 떄 format을 위와 같이 지정하고, 로그 수준을 INFO로 하면 word2vec의 학습과정에서 로그 메시지를 양식에 맞게 INFO 수준으로 볼 수 있다.


```python
from gensim.models import word2vec

print("Training model...")

model = word2vec.Word2Vec(sentences,
                        workers=num_workers,
                        size=num_features,
                        min_count=min_word_count,
                        window=context,
                        sample=downsampling)
```

    2019-07-12 21:56:43,872 : INFO : collecting all words and their counts
    2019-07-12 21:56:43,872 : INFO : PROGRESS: at sentence #0, processed 0 words, keeping 0 word types
    

    Training model...
    

    2019-07-12 21:56:44,092 : INFO : PROGRESS: at sentence #10000, processed 1205223 words, keeping 51374 word types
    2019-07-12 21:56:44,305 : INFO : PROGRESS: at sentence #20000, processed 2396605 words, keeping 67660 word types
    2019-07-12 21:56:44,406 : INFO : collected 74065 word types from a corpus of 2988089 raw words and 25000 sentences
    2019-07-12 21:56:44,406 : INFO : Loading a fresh vocabulary
    2019-07-12 21:56:44,443 : INFO : min_count=40 retains 8160 unique words (11% of original 74065, drops 65905)
    2019-07-12 21:56:44,443 : INFO : min_count=40 leaves 2627273 word corpus (87% of original 2988089, drops 360816)
    2019-07-12 21:56:44,454 : INFO : deleting the raw counts dictionary of 74065 items
    2019-07-12 21:56:44,469 : INFO : sample=0.001 downsamples 30 most-common words
    2019-07-12 21:56:44,469 : INFO : downsampling leaves estimated 2494384 word corpus (94.9% of prior 2627273)
    2019-07-12 21:56:44,481 : INFO : estimated required memory for 8160 words and 300 dimensions: 23664000 bytes
    2019-07-12 21:56:44,481 : INFO : resetting layer weights
    2019-07-12 21:56:44,577 : INFO : training model with 4 workers on 8160 vocabulary and 300 features, using sg=0 hs=0 sample=0.001 negative=5 window=10
    2019-07-12 21:56:45,578 : INFO : EPOCH 1 - PROGRESS: at 53.88% examples, 1354613 words/s, in_qsize 7, out_qsize 0
    2019-07-12 21:56:46,589 : INFO : EPOCH 1 - PROGRESS: at 97.12% examples, 1208841 words/s, in_qsize 8, out_qsize 0
    2019-07-12 21:56:46,625 : INFO : worker thread finished; awaiting finish of 3 more threads
    2019-07-12 21:56:46,631 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-07-12 21:56:46,639 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-07-12 21:56:46,645 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-07-12 21:56:46,649 : INFO : EPOCH - 1 : training on 2988089 raw words (2494785 effective words) took 2.1s, 1208654 effective words/s
    2019-07-12 21:56:47,643 : INFO : EPOCH 2 - PROGRESS: at 51.68% examples, 1296637 words/s, in_qsize 7, out_qsize 0
    2019-07-12 21:56:48,565 : INFO : worker thread finished; awaiting finish of 3 more threads
    2019-07-12 21:56:48,565 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-07-12 21:56:48,579 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-07-12 21:56:48,587 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-07-12 21:56:48,587 : INFO : EPOCH - 2 : training on 2988089 raw words (2494872 effective words) took 1.9s, 1287927 effective words/s
    2019-07-12 21:56:49,584 : INFO : EPOCH 3 - PROGRESS: at 52.03% examples, 1307272 words/s, in_qsize 7, out_qsize 0
    2019-07-12 21:56:50,495 : INFO : worker thread finished; awaiting finish of 3 more threads
    2019-07-12 21:56:50,504 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-07-12 21:56:50,504 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-07-12 21:56:50,513 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-07-12 21:56:50,513 : INFO : EPOCH - 3 : training on 2988089 raw words (2494507 effective words) took 1.9s, 1297134 effective words/s
    2019-07-12 21:56:51,508 : INFO : EPOCH 4 - PROGRESS: at 52.60% examples, 1319590 words/s, in_qsize 7, out_qsize 0
    2019-07-12 21:56:52,421 : INFO : worker thread finished; awaiting finish of 3 more threads
    2019-07-12 21:56:52,438 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-07-12 21:56:52,438 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-07-12 21:56:52,444 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-07-12 21:56:52,444 : INFO : EPOCH - 4 : training on 2988089 raw words (2494292 effective words) took 1.9s, 1293999 effective words/s
    2019-07-12 21:56:53,445 : INFO : EPOCH 5 - PROGRESS: at 54.20% examples, 1360524 words/s, in_qsize 7, out_qsize 0
    2019-07-12 21:56:54,212 : INFO : worker thread finished; awaiting finish of 3 more threads
    2019-07-12 21:56:54,228 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-07-12 21:56:54,235 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-07-12 21:56:54,241 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-07-12 21:56:54,242 : INFO : EPOCH - 5 : training on 2988089 raw words (2494012 effective words) took 1.8s, 1390220 effective words/s
    2019-07-12 21:56:54,242 : INFO : training on a 14940445 raw words (12472468 effective words) took 9.7s, 1291189 effective words/s
    

word2vec으로 학습시킨 모델의 경우 모델을 따로 저장해두면 이후 다시 사용할수 있기 때문에 저장해두자


```python
# 모델의 하이퍼파라미터를 설정한 내용을 모델 이름에 담는다면 나중에 참고하기 용이하다.
# 모델을 저장하면 Word2Vec.load()를 통해 다시 사용할 수 있다.
model_name = "300features_40minwords_10context"
model.save(model_name)
```

    2019-07-12 22:00:40,807 : INFO : saving Word2Vec object under 300features_40minwords_10context, separately None
    2019-07-12 22:00:40,807 : INFO : not storing attribute vectors_norm
    2019-07-12 22:00:40,807 : INFO : not storing attribute cum_table
    C:\Users\nicey\.conda\envs\nlp\lib\site-packages\smart_open\smart_open_lib.py:398: UserWarning: This function is deprecated, use smart_open.open instead. See the migration notes for details: https://github.com/RaRe-Technologies/smart_open/blob/master/README.rst#migrating-to-the-new-open-function
      'See the migration notes for details: %s' % _MIGRATION_NOTES_URL
    2019-07-12 22:00:41,229 : INFO : saved 300features_40minwords_10context
    

에제 만들어진 word2vec 모델을 활용해 선형 회귀 모델을 학습해보자. 우선 학습을 위해 하나의 리뷰를 같은 형태의 입력값으로 만들어야 한다. 지금은 word2vec 모델에서 각 단어가 벡터로 표현돼 있다. 그리고 리뷰마다 단어의 개수가 모두 다르기 때문에 입력값을 하나으 형태로 만들어야 한다. 가장 단순한 방법은 문장에 있는 모든 단어의 벡터값에 대해 평균을 내서 리뷰 하나당 하나의 벡터로 만드는 방법이 있다.

그럼 하나의 리뷰에 대해 전체 단어의 평균값을 계산하는 함수를 구현하자


```python
import numpy as np
```


```python
def get_features(words, model, num_features):
    # 출력 벡터 초기화
    feature_vector = np.zeros((num_features), dtype=np.float32)
    
    num_words = 0
    # 어휘 사전 준비
    index2word_set = set(model.wv.index2word)
    
    for w in words:
        if w in index2word_set:
            num_words = 1
            # 사전에 해당하는 단어에 대해 단어 벡터를 더함
            feature_vector = np.add(feature_vector, model[w])
            
    # 문장의 단어 수만큼 나누어 단어 벡터의 평균값을 문장 벡터로 함
    feature_vector = np.divide(feature_vector, num_words)
    return feature_vector
```

- words : 단어의 모음인 하나의 리뷰가 들어간다.
- model : word2vec 모델을 넣는 공이며, 우리가 학습한 word2vec 모델이 들어간다.
- num_features : word2vec으로 임베딩할 때 정했던 벡터의 차원 수를 뜻한다.

하나의 벡터를 만드는 과정을 빠르게 하기 위해 np.zeros를 사용해 미리 모두 0값을 가지는 벡터를 만든다. 그리고 문장의 단어가 모델 단어사전에 속하는지 보기 위해 model.wv.index2word를 set객체로 생성해서 index2word_set 변수에 할당한다. 다음 반복문을 통해 리뷰를 구성하는 단어에 대해 임베딩된 벡터가 있는 단어 벡터의 합을 구하고 사용한 단어의 전체 개수로 나누어 평균 벡터의 값을 구한다.

문장에 특징값을 만들 수 있는 함수를 구현했다면 이제 앞에서 정의한 함수를 사용해 전체 리뷰에 대해 각 리뷰의 평균 벡터를 구하는 함수를 정의하자


```python
def get_dataset(reviews, model, num_features):
    dataset = list()
    
    for s in reviews:
        dataset.append(get_features(s, model, num_features))
        
    reviewFeatureVecs = np.stack(dataset)
    return reviewFeatureVecs
```

- reviews : 학습 데이터인 전체 리뷰 데이터를 입력
- model : word2vec 모델을 입력
- num_features : word2vec으로 임베딩할 때 정했던 벡터의 차원 수

전체 리뷰에 대한 평균 벡터를 담을 0으로 채워진 numpy 배열을 미리 만든다. 배열은 2차원, 배열의 행에는 각 문장에 대한 길이, 열에는 평균 벡터의 차원수 즉 크기를 입력. 그리고 각 리뷰에 대해 반복문을 돌면서 각 리뷰에 대해 특징 값을 만든다.

구현한 함수를 사용해 실제 학습에 사용될 입력값을 만들어 보자.


```python
test_data_vecs = get_dataset(sentences, model, num_features)
```

    C:\Users\nicey\.conda\envs\nlp\lib\site-packages\ipykernel_launcher.py:13: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
      del sys.path[0]
    

### 학습과 검증 데이터셋 분리


```python
from sklearn.model_selection import train_test_split
import numpy as np

X = test_data_vecs
y = np.array(sentiments)

RANDOM_SEED = 42
TEST_SPLIT = 0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, 
                                                    random_state=RANDOM_SEED)

```

### 모델 선언 및 학습


```python
from sklearn.linear_model import LogisticRegression

lgs = LogisticRegression(class_weight='balanced')
lgs.fit(X_train, y_train)
```

    C:\Users\nicey\.conda\envs\nlp\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    




    LogisticRegression(C=1.0, class_weight='balanced', dual=False,
                       fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                       max_iter=100, multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False)



class_weight을 `balanced`로 설정했다. 이는 각 라벨에 대해 균형있게 학습하기 위함이다.

### 검증 데이터셋을 이용한 성능 평가


```python
predicted = lgs.predict(X_test)
from sklearn import metrics

fpr, tpr, _ = metrics.roc_curve(y_test, (lgs.predict_proba(X_test)[:, 1]))
auc = metrics.auc(fpr, tpr)

print("------------")
print("Accuracy: %f" % lgs.score(X_test, y_test))  #checking the accuracy
print("Precision: %f" % metrics.precision_score(y_test, predicted))
print("Recall: %f" % metrics.recall_score(y_test, predicted))
print("F1-Score: %f" % metrics.f1_score(y_test, predicted))
print("AUC: %f" % auc)
```

    ------------
    Accuracy: 0.874200
    Precision: 0.869141
    Recall: 0.883287
    F1-Score: 0.876157
    AUC: 0.940927
    

학습 결과를 확인해 보면 TF-IDF를 사용해서 학습한 것보다 상대적으로 성능이 조금 떨어지는 것을 볼 수 있다. word2vec이 단어 간의 유사도를 보는 관점에서는 분명히 효과적일 수는 있지만 항상 좋은 성능을 보장하지는 않는다는 점을 알 수 있다.

## 데이터 제출


```python
TEST_CLEAN_DATA = 'test_clean.csv'

test_data = pd.read_csv(DATA_IN_PATH + TEST_CLEAN_DATA)

test_review = list(test_data['review'])
```


```python
test_data.head(5)
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




```python
test_sentences = list()

for review in test_review:
    test_sentences.append(review.split())
```


```python
test_data_vecs = get_dataset(test_sentences, model, num_features)
test_predicted = lgs.predict(test_data_vecs)
```

    C:\Users\nicey\.conda\envs\nlp\lib\site-packages\ipykernel_launcher.py:13: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
      del sys.path[0]
    


```python
import os

DATA_OUT_PATH = './data_out/'

if not os.path.exists(DATA_OUT_PATH):
    os.makedirs(DATA_OUT_PATH)

ids = list(test_data['id'])

answer_dataset = pd.DataFrame({'id': ids, 'sentiment': test_predicted})

answer_dataset.to_csv(DATA_OUT_PATH + 'lgs_w2v_answer.csv', index=False, quoting=3)
```