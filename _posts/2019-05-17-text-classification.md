---
layout: post
comments: true
title: '[NLP] 텍스트 분류 (실습)'
categories: [bigdata]
tags: [ml]
date: 2019-06-18
---

### 텍스트 분류

캐글(Kaggle)의 "Bag of Words with bag of popcorns"

### 워드팝콘 데이터의 처리 과정

> 케글 데이터 불러오기 -> EDA -> 데이터정제 -> 모델링

**데이터정제**
- HTML 및 문장 보호 제거
- 불용어 제거
- 단어 최대 길이 설정
- 단어 패딩
- 백터 표상화

```python
import zipfile
```


```python
DATA_IN_PATH = './data_in/'
```


```python
file_list = ['labeledTrainData.tsv.zip', 'unlabeledTrainData.tsv.zip', 'testData.tsv.zip']
file_list_tsv = ['labeledTrainData.tsv', 'unlabeledTrainData.tsv', 'testData.tsv']
```


```python
! ls ./data_in
```

    NanumGothic.ttf		  ratings.txt		test_id.npy
    labeledTrainData.tsv	  ratings_test.txt	train_clean.csv
    labeledTrainData.tsv.zip  ratings_train.txt	train_input.npy
    nsmc_test_input.npy	  sampleSubmission.csv	train_label.npy
    nsmc_test_label.npy	  testData.tsv		unlabeledTrainData.tsv
    nsmc_train_input.npy	  testData.tsv.zip	unlabeledTrainData.tsv.zip
    nsmc_train_label.npy	  test_clean.csv



```python
# file_list의 zip 압축파일을 풀어준다.
for file in file_list:
    zipRef = zipfile.ZipFile(DATA_IN_PATH + file, 'r')
    zipRef.extractall(DATA_IN_PATH)
    zipRef.close()
```


```python
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
```


```python
! head -2 ./data_in/labeledTrainData.tsv
```

    id	sentiment	review
    "5814_8"	1	"With all this stuff going down at the moment with MJ i've started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ's feeling towards the press and also the obvious message of drugs are bad m'kay.<br /><br />Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring. Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him.<br /><br />The actual feature film bit when it finally starts is only on for 20 minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord. Why he wants MJ dead so bad is beyond me. Because MJ overheard his plans? Nah, Joe Pesci's character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno, maybe he just hates MJ's music.<br /><br />Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence. Also, the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene.<br /><br />Bottom line, this movie is for people who like MJ on one level or another (which i think is most people). If not, then stay away. It does try and give off a wholesome message and ironically MJ's bestest buddy in this movie is a girl! Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty? Well, with all the attention i've gave this subject....hmmm well i don't know because people can be different behind closed doors, i know this for a fact. He is either an extremely nice but stupid guy or one of the most sickest liars. I hope he is not the latter."



```python
# 데이터를 불러오자
train_data = pd.read_csv(DATA_IN_PATH + 'labeledTrainData.tsv', header=0, delimiter="\t")
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
      <th>id</th>
      <th>sentiment</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5814_8</td>
      <td>1</td>
      <td>With all this stuff going down at the moment w...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2381_9</td>
      <td>1</td>
      <td>\The Classic War of the Worlds\" by Timothy Hi...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7759_3</td>
      <td>0</td>
      <td>The film starts with a manager (Nicholas Bell)...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3630_4</td>
      <td>0</td>
      <td>It must be assumed that those who praised this...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9495_8</td>
      <td>1</td>
      <td>Superbly trashy and wondrously unpretentious 8...</td>
    </tr>
  </tbody>
</table>
</div>



데이터는 "id", "sentiment", "review"로 구분되어 있어며, 각 리뷰('review')에 대한 감정('sentiment')이 긍정(1), 부정(0)인지 나와있다.

이제 다음과 같은 순서로 데이터 분석을 해보자.

1. 데이터 크기
2. 데이터 개수
3. 각 리뷰 문자 길이 분포
4. 많이 사용된 단어
5. 긍정, 부정 데이터의 분포
6. 각 리뷰의 단어 개수 분포
7. 특수문자 및 대문자, 소문자 비율

### 데이터 크기


```python
print("파일 크기 :")
for file in os.listdir(DATA_IN_PATH):
    if file in file_list_tsv:
        print(file.ljust(30) + str(round(os.path.getsize(DATA_IN_PATH + file) / 1000000, 2)) + 'MB')
```

    파일 크기 :
    labeledTrainData.tsv          33.56MB
    testData.tsv                  32.72MB
    unlabeledTrainData.tsv        67.28MB


### 데이터 개수


```python
print('전체 학습 데이터의 개수: {}'.format(len(train_data)))
```

    전체 학습 데이터의 개수: 25000


### 각 리뷰의 문자 길이 분포


```python
train_length = train_data['review'].apply(len)
train_length.head()
```




    0    2302
    1     946
    2    2449
    3    2245
    4    2231
    Name: review, dtype: int64




```python
plt.figure(figsize=(12,5))
plt.hist(train_length, bins=200, alpha=0.5, color='r', label='word')
plt.yscale('log', nonposy='clip')
plt.title('Log-Histogram of length of review')
plt.xlabel('Length of review')
plt.ylabel('Number of review')
```




    Text(0, 0.5, 'Number of review')




![png](/assets/img/post/nlp-text-classification/output_18_1.png)


- figsize: (가로, 세로) 형태의 튜플로 입력
- bins: 히스토그램 값들에 대한 버켓 범위
- range: x축 값의 범위
- alpha: 그래프 색상 투명도
- color: 그래프 색상
- label: 그래프에 대한 라벨

대부분 6000 이하 그중에서도 2000 이하에 분포되어 있음을 알 수 있다. 그리고 일부 데이터의 경우 이상치로 10000 이상의 값을 가지고 있다.

길이데 대해 몇가지 통계를 확인해 보자


```python
print('리뷰 길이 최대 값: {}'.format(np.max(train_length)))
print('리뷰 길이 최소 값: {}'.format(np.min(train_length)))
print('리뷰 길이 평균 값: {:.2f}'.format(np.mean(train_length)))
print('리뷰 길이 표준편차: {:.2f}'.format(np.std(train_length)))
print('리뷰 길이 중간 값: {}'.format(np.median(train_length)))
# 사분위의 대한 경우는 0~100 스케일로 되어있음
print('리뷰 길이 제 1 사분위: {}'.format(np.percentile(train_length, 25)))
print('리뷰 길이 제 3 사분위: {}'.format(np.percentile(train_length, 75)))
```

    리뷰 길이 최대 값: 13708
    리뷰 길이 최소 값: 52
    리뷰 길이 평균 값: 1327.71
    리뷰 길이 표준편차: 1005.22
    리뷰 길이 중간 값: 981.0
    리뷰 길이 제 1 사분위: 703.0
    리뷰 길이 제 3 사분위: 1617.0


리뷰의 길이가 히스토그램에서 확인했던 것과 비슷하게 평균이 1300정도 이고, 최댓값이 13000이로 확인이 된다.


```python
plt.figure(figsize=(12,5))
plt.boxplot(train_length, labels=['counts'], showmeans=True)
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x7f1694cad4a8>,
      <matplotlib.lines.Line2D at 0x7f1694cad7f0>],
     'caps': [<matplotlib.lines.Line2D at 0x7f1694cadb38>,
      <matplotlib.lines.Line2D at 0x7f1694cade80>],
     'boxes': [<matplotlib.lines.Line2D at 0x7f1694cad358>],
     'medians': [<matplotlib.lines.Line2D at 0x7f1694902208>],
     'fliers': [<matplotlib.lines.Line2D at 0x7f1694902860>],
     'means': [<matplotlib.lines.Line2D at 0x7f1694902550>]}




![png](/assets/img/post/nlp-text-classification/output_22_1.png)


- labels : 입력한 데이터에 대한 라벨
- showmeans : 평균값을 마크함

데이터의 길이가 대부분 2000 이하로 평균이 1500 이하인데, 길이가 4000 이상인 이상치 데이터도 많이 분포되어 있는 것을 확인할 수 있다.

이제 리뷰에서 많이 사용된 단어로 어떤 것이 있는지 알아보자.

### 많이 사용된 단어


```python
from wordcloud import WordCloud
cloud = WordCloud(width=800, height=600).generate(" ".join(train_data['review']))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
```




    (-0.5, 799.5, 599.5, -0.5)




![png](/assets/img/post/nlp-text-classification/output_25_1.png)


워드 클라우드를 통해 살펴보면 가장 많이 사용된 단어는 br로 확인이 된다. br은 HTML 태그 이기때문에 이 태그들을 모두 제거 하는 전처리 작업이 필요하다.

### 긍정/부정 데이터의 분포


```python
fig, axe = plt.subplots(ncols=1)
fig.set_size_inches(6, 3)
sns.countplot(train_data['sentiment'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f1694a64898>




![png](/assets/img/post/nlp-text-classification/output_28_1.png)


거의 동일한 개수로 분포되어 있음을 확인할 수 있다. 좀더 정확한 값을 확인해 보자.


```python
print("긍정 리뷰 개수: {}".format(train_data['sentiment'].value_counts()[1]))
print("부정 리뷰 개수: {}".format(train_data['sentiment'].value_counts()[0]))
```

    긍정 리뷰 개수: 12500
    부정 리뷰 개수: 12500


### 각 리뷰의 단어 개수 분포

각 리뷰를 단어 기준으로 나눠서 각 리뷰당 단어의 개수를 확인해 보자. 
단어는 띄어쓰기 기준으로 하나의 단어라 생각하고 개수를 계산한다. 
우선 각 단어의 길이를 가지는 변수를 하나 설정하자.


```python
train_word_counts = train_data['review'].apply(lambda x:len(x.split(' ')))
```


```python
plt.figure(figsize=(15, 10))
plt.hist(train_word_counts, bins=50, facecolor='r',label='train')
plt.title('Log-Histogram of word count in review', fontsize=15)
plt.yscale('log', nonposy='clip')
plt.legend()
plt.xlabel('Number of words', fontsize=15)
plt.ylabel('Number of reviews', fontsize=15)
```




    Text(0, 0.5, 'Number of reviews')




![png](/assets/img/post/nlp-text-classification/output_34_1.png)


대부분의 단어가 1000개 미만의 단어를 가지고 있고, 대부분 200개 정도의 단어를 가지고 있다.


```python
print('리뷰 단어 개수 최대 값: {}'.format(np.max(train_word_counts)))
print('리뷰 단어 개수 최소 값: {}'.format(np.min(train_word_counts)))
print('리뷰 단어 개수 평균 값: {:.2f}'.format(np.mean(train_word_counts)))
print('리뷰 단어 개수 표준편차: {:.2f}'.format(np.std(train_word_counts)))
print('리뷰 단어 개수 중간 값: {}'.format(np.median(train_word_counts)))
# 사분위의 대한 경우는 0~100 스케일로 되어있음
print('리뷰 단어 개수 제 1 사분위: {}'.format(np.percentile(train_word_counts, 25)))
print('리뷰 단어 개수 제 3 사분위: {}'.format(np.percentile(train_word_counts, 75)))
```

    리뷰 단어 개수 최대 값: 2470
    리뷰 단어 개수 최소 값: 10
    리뷰 단어 개수 평균 값: 233.79
    리뷰 단어 개수 표준편차: 173.74
    리뷰 단어 개수 중간 값: 174.0
    리뷰 단어 개수 제 1 사분위: 127.0
    리뷰 단어 개수 제 3 사분위: 284.0


통계를 살펴보면 평균이 233개, 최댓값은 2470개의 단어를 가지고 있다. 그리고 3사분위 값이 284개로 리뷰의 75%가 300개 이하의 단어를 가지고 있음을 확인 할 수 있다.

### 특수문자 및 대/소문자 비율


```python
qmarks = np.mean(train_data['review'].apply(lambda x: '?' in x)) # 물음표가 구두점으로 쓰임
fullstop = np.mean(train_data['review'].apply(lambda x: '.' in x)) # 마침표
capital_first = np.mean(train_data['review'].apply(lambda x: x[0].isupper())) #  첫번째 대문자
capitals = np.mean(train_data['review'].apply(lambda x: max([y.isupper() for y in x]))) # 대문자가 몇개
numbers = np.mean(train_data['review'].apply(lambda x: max([y.isdigit() for y in x]))) # 숫자가 몇개
                  
print('물음표가있는 질문: {:.2f}%'.format(qmarks * 100))
print('마침표가 있는 질문: {:.2f}%'.format(fullstop * 100))
print('첫 글자가 대문자 인 질문: {:.2f}%'.format(capital_first * 100))
print('대문자가있는 질문: {:.2f}%'.format(capitals * 100))
print('숫자가있는 질문: {:.2f}%'.format(numbers * 100))
```

    물음표가있는 질문: 29.55%
    마침표가 있는 질문: 99.69%
    첫 글자가 대문자 인 질문: 92.84%
    대문자가있는 질문: 99.59%
    숫자가있는 질문: 56.66%


결과를 보면 대부분 마침표를 포함하고 있고, 대문자도 대부분 사용하고 있다. 따라서 전처리 과정에서 대문자의 경우 모두 소문자로 바꾸고 특수 문자의 경우 제거해야 한다. 이 과정은 학습에 방해가 되는 요소들을 제거하기 위함이다.
