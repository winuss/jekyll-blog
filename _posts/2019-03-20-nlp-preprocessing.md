---
layout: post
comments: true
title: '[NLP] 자연어처리 - 데이터 정제'
categories: [bigdata]
tags: [ml]
date: 2019-03-20
---

![nlp-pre](/assets/img/post/nlp-preprocessing/nlp-pre.png)

# Data Cleaning and Text Preprocessing

기계가 텍스트를 이해할 수 있도록 텍스트를 정제하고 신호와 소음을 구분하여 아웃라이어 데이터로 인한 오버피팅을 방지하기 위해서는 다음과 같은 처리를 해주어야 한다.

- HTML 태그, 특수문자, 이모티콘 처리
- 토근화(Tokenization) : 문장의 단어를 분리하는 단계
- 불용어(Stopword) 제거 : 자주 등장하지만 특별한 의미를 갖지 않는 단어 제거
- 어간 추출(Stemming) 및 음소표기법(Lemmatization)
- 정규 표현식

>**텍스트 데이터 전처리 이해**<br><br>
>`정규화 normalization` (입니닼ㅋㅋ -> 입니다 ㅋㅋ, 샤릉해 -> 사랑해)<br>
> 한국어를 처리하는 예시입니닼ㅋㅋㅋㅋㅋ -> 한국어를 처리하는 예시입니다 ㅋㅋ<br><br>
>`토큰화 tokenization`<br>
>한국어를 처리하는 예시입니다 ㅋㅋ -> 한국어Noun, 를Josa, 처리Noun, 하는Verb, 예시Noun, 입Adjective, 니다Eomi ㅋㅋKoreanParticle<br><br>
>`어근화 stemming` (입니다 -> 이다)<br>
>한국어를 처리하는 예시입니다 ㅋㅋ -> 한국어Noun, 를Josa, 처리Noun, 하다Verb, 예시Noun, 이다Adjective, ㅋㅋKoreanParticle<br><br>
>`어근 추출 phrase extraction`<br>
>한국어를 처리하는 예시입니다 ㅋㅋ -> 한국어, 처리, 예시, 처리하는 예시<br><br>
>(출처 : [트위터 한국어 형태소 분석기](https://github.com/twitter/twitter-korean-text))

# BeautifulSoup

많이 쓰이는 파이썬용 파서로 html, xml을 파싱할때 주로 많이 사용한다.

BeautifulSoup에는 기본적으로 파이썬 표준 라이브러리인 html파서를 지원하지만, lxml을 사용하면 성능 향상이 있다.

**BeautifulSoup(markup, `'[파서명]'`)**

- html.parser : 빠르지만 유연하지 않기 때문에 단순한 html문서에 사용
- lxml : 매우 빠르고 유연
- xml : xml 파일에 사용
- html5lib : 복접한 구조의 html에 대해서 사용.(속도가 느린편)

~~~python
from bs4 import BeautifulSoup
import re

data = "<p>초등학교 입학을 축하합니다.~~~^^;<br/></p>"
soup = BeautifulSoup(data, "html5lib")
remove_tag = soup.get_text()
result_text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》;]', ''
              , remove_tag)
print(result_text)
~~~

~~~result
결과 : 초등학교 입학을 축하합니다.
~~~

`<p>초등학교 입학을 축하합니다.~~~^^;<br/></p>` 문장에서 태그와 특수문자를 제거하기 위해 BeautifulSoup 와 정규표현식을 사용하였다.

# 토큰화 (Tokenization)

코퍼스(corpus)에서 토큰(token)이라 불리는 단위로 나누는 작업을 토큰화(Tokenization)라고 부른다. 토큰의 단위가 상황에 따라 다르지만, 보통 의미있는 단위로 토큰 정의한다.

토큰의 기준을 단어로 하는 경우 가장 간단한 방법은 띄어쓰기를 기준으로 자르는 것이다.

**I loved you. machine learning** 에서 구두점을 제외시키고 토큰화 한 결과는 다음과 같다.

`"I", "loved", "you", "machine", "learning"`

하지만 보통 토큰화 작업은 단순히 구두점이나 특수문자를 전부 제거하는 작업을 수행하는 것만으로 해결되지 않는다. 구두점이나 특수문자를 전부 제거하면 토큰이 의미를 잃어 버리는 경우가 발생하기도 하기때문이다. 띄어쓰기 단위로 자르면 사실상 단어 토큰이 구분되는 영어와 달리, 한국어는 띄어쓰기반으로는 단어 토큰을 구분하기 어렵다. 

>한국어는 단어의 다양한 의미와 높낮이 그리고 띄어쓰기가 어렵다 보니 잘못된 데이터를 많이 받게 되어 자연어처리를 하는데 있어 어려움이 많이 따른다. 하지만 다양한 곳에서 한국어 처리를 위한 형태소 분석기를 연구하고 있다. 얼마전 카카오에서도 카이라는 딥러닝 기술 기반의 형태소 분석기를 오픈소스로 공개 하였고 그외에 트위터, 코엔엘파이등 꽤 쓸만한 것들이 있다.


# 불용어 (Stopword)

일반적으로 코퍼스에서 자주 나타나는 단어로 학습이나 예측 프로세스에 실제로 기여를 하지 않는다.

(조사, 접미사 - 나,나,은,는,이,가,하다,합니다....등)

NLTK에는 17개의 언어에 대해 불용어가 정의되어 있다. 하지만 아쉽게도 한국어는...없다.

간단하게 10개 정도만 영어의 불용어를 보면,

~~~python
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords.words('english')[:10]
~~~
>결과 : ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your']

간단하다, 불용어 사전에 있는 단어들은 제거하면 된다.

한국어의 경우 불용어를 제거하는 방법으로는 위에서 언급한 형태소 분석 후 조사, 접속사 등을 제거할 수 있다. 

# 어간 추출 (Stemming)

스태밍이라고 하는데, 어간이란 말은 단어의 의미를 담고 있는 **단어의 핵심부분**이라 생각하면 된다. 쉽게, 단어를 축약형으로 바꿔주는 작업이라고도 할 수 있다.

한국어가, 한국어는, 한국어처럼 -> `한국어`

대표적으로 포터 스태머(PorterStemmer)와 랭커스터 스태머(LancasterStemmer)가 있는데 포터는 보수적이고 랭커스터는 좀 더 적극적이다.

**PorterStemmer**
~~~python
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
print(stemmer.stem('maximum'))
print("running >> {}".format(stemmer.stem("running")))
print("runs >> {}".format(stemmer.stem("runs")))
print("run >> {}".format(stemmer.stem("run")))
~~~

>maxim<br>
>running >> run<br>
>runs >> run<br>
>run >> run

**LancasterStemmer**
~~~python
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
print(lancaster_stemmer.stem('maximum'))
print("running >> {}".format(lancaster_stemmer.stem("running")))
print("runs >> {}".format(lancaster_stemmer.stem("runs")))
print("run >> {}".format(lancaster_stemmer.stem("run")))
~~~

>maxim<br>
>running >> run<br>
>runs >> run<br>
>run >> run


# 음소표기법 (Lemmatization)

언어학에서 음소표기법 (Lemmatization)은 단어의 보조 정리 또는 사전 형식에 의해 식별되는 단일 항목으로 분석 될 수 있도록 굴절 된 형태의 단어를 그룹화하는 과정이다.
어간 추출(Stemming)과는 달리 단어의 형태가 적절히 보존되는 양상을 보이는 특징이 있다. 하지만 그럼에도 의미를 알 수 없는 적절하지 못한 단어로 변환 하기도 하는데 음소표기법(Lemmatizer)은 본래 단어의 품사 정보를 알아야만 정확한 결과를 얻을 수 있기 때문이다.

- 품사정보가 보존된 형태의 기본형으로 변환. 
- 단어가 명사로 쓰였는지 동사로 쓰였는지에 따라 적합한 의미를 갖도록 추출하는 것.

~~~python
from nltk.stem import WordNetLemmatizer
n=WordNetLemmatizer()
words=['have', 'going', 'love', 'lives', 'fly', 'dies', 'has', 'starting']
[n.lemmatize(w) for w in words]
~~~

>결과 : ['have', 'going', 'love', 'life', 'fly', 'dy', 'ha', 'starting']

결과에서 보면 알수 있듯이 `dy`나 `ha`는 그 의미를 알수 없는 적절하지 못한 단어로 변환이 되었다.


하지만 dies나 has가 동사로 쓰였다는 것을 알려준다면 좀더 정확한 결과를 얻을 수 있게 된다.
~~~python
n.lemmatize('dies', 'v')
~~~

> 'die'

~~~python
n.lemmatize('has', 'v')
~~~

> 'have'

음소표기법은 문맥을 고려하며, 수행했을 때의 결과는 해당 단어의 품사 정보를 보존한다. 하지만 어간 추출은 품사 정보가 보존이 되지 않는다.

# 마치며..

이런 작업들이 갖고있는 의미는 눈으로 봤을 때는 서로 다른 단어들이지만, 하나의 단어로 일반화시킬 수 있다면 하나의 단어로 일반화시켜서 문서 내의 단어 수를 줄여보자는 것이고 자연어 처리에서 전처리의 지향점은 언제나 갖고 있는 코퍼스로부터 복잡성을 줄이는 일이다.