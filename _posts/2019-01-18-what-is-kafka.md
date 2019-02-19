---
layout: post
comments: true
title: '카프카(Kafka)의 이해'
categories: [bigdata]
tags: [datalake]
date: 2019-01-18
---

![img](/assets/img/post/post-kafka/kafka-1.png)

대용량 게임로그 수집을 위해 Elastic Stack을 도입하게 되었는데, 중간에 버퍼역할(메시지큐)을 하는 Kafka에 대서 알아보려고 한다.

### 메시지큐?

메시지 지향 미들웨어(Message Oriented Middleware: MOM)은 비동기 메시지를 사용하는 다른 응용프로그램 사이의 데이터 송수신을 의미하는데 MOM을 구현한 시스템을 메시지큐(Message Queue:MQ)라 한다.

### 카프카란?

분산형 스트리밍 플랫폼(A distributed streaming platform)이다. LinkedIn에서 여러 구직 및 채용 정보들을 한곳에서 처리(발행/구독)할 수 있는 플래폼으로 개발이 시작 되었다고 한다.

(발행/구독: pub-sub은 메시지를 특정 수신자에게 직접적으로 보내주는 시스템이 아니고, 메시지를 받기를 원하는 사람이 해당 토픽(topic)을 구독함으로써 메시지를 읽어 올 수 있다.)



#### 카프카의 특징

대용량 실시간 로그처리에 특화되어 설계된 메시징 시스템으로 TPS가 매우 우수하고,

메시지를 메모리에 저장하는 기존 메시징 시스템과는 달리 파일에 저장을 하는데 그로 인해 카프카를 재시작해도 메시지 유실 우려가 감소된다.

기본 메시징 시스템(rabbitMQ, ActiveMQ)에서는 브로커(Broker)가 컨슈머(consumer)에게 메시지를 push해 주는 방식인데, 카프카는 컨슈머(Consumer)가 브로커(Broker)로부터 메시지를 직접 가져가는 PULL 방식으로 동작하기 때문에 컨슈머는 자신의 처리 능력만큼의 메시지만 가져와 최적의 성능을 낼 수 있다. 대용량처리에 특화 되었다는 것은 아마도 이러한 구조로 설계가 되어 가능하게 된게 아닌가 싶다.

![img](/assets/img/post/post-kafka/kafka-2.png)

여기서 한가지 의문이 든다. 일반적으로 파일보다 메모리가 성능이 우수한데 왜 카프카가 성능이 좋은 것일까? 그 이유는 카프카의 파일 시스템을 활용한 고성능 디자인에 있다. 일반적으로 하드디스크는 메모리보다 수백배 느리지만 하드디스크의 순차적 읽기에 대한 성능은 메모리보다 크게 떨어지지 않는다고 한다. 



컨슈머(Consumer)와 브로커(Broker)에 대해서는 카프카 구성요소에 대한 설명에서 좀 더 자세히 알아보자.



#### 카프카의 구성요소

카프카에는 다음과 같이 여러 구성요소가 있다.

**topic, partition, offset**

**producer, consumer, consumer group**

**broker, zookeeper**

**replication**

구성요소 하나씩 살펴보도록 하자.



**Topic, Partition** : 카프카에 저장되는 메시지는 topic으로 분류되고, topic은 여러개의 patition으로 나눠질수 있다. partition안에는 message의 상대적 위치를 내타내는 offset이 있는데 이 offet정보를 이용해 이전에 가져간 메시지의 위치 정보를 알 수 있고 동시에 들어오는 많은 데이터를 여러개의 파티션에 나누어 저장하기 때문에 병렬로 빠르게 처리할 수 있다.

![img](/assets/img/post/post-kafka/kafka-3.png)

**Producer, Consumer** : 말대로 Producer는 생산(메시지를 Write)하는 주체, Consumer는 소비(메시지를 Read)하는 주체이다. Producer와 Consumer간에는 상호 존재 여부를 알지 못한채 자신에게 주어진 역할만 처리 하게 된다. (위 그림에서 보면 Writes가 Producer)

**Consumer Group** : Producer에서 생산(Write)한 메시지는 여러개의 파티션에 저장을 하는데, 그렇다면 소비하는(Consumer)하는 쪽에서도 여러 소비자가 메시지를 읽어가는것이 훨씬 효율적일 것이다. 하나의 목표를 위해 소비를 하는 그룹, 즉 하나의 토픽을 읽어가기 위한 Counsumer들을 Consumer Group라고 한다.

하지만 이 Consumer Group에는 한가지 룰이 있다. Topic의 파티션은 그 Consumer Group과 1:n 매칭. 즉, 자신이 읽고 있는 파티션에는 같은 그룹내 다른 컨슈머가 읽을 수 없다. (파티션에는 동일한 Consumer Group을 위한 하나의 구멍이 있고, Consumer는 그 구멍에 빨대를 꽂아 읽어간다고 생각하면 쉽게 상상이 될지도….^^;)

![img](/assets/img/post/post-kafka/kafka-4.png)

위 그림과 같이 하나의 토픽에 4개의 파티션이 있고 컨슈머그룹내 3개의 컨슈머가 있다면 컨슈머1,2,3은 각 파티션 1,2,3에 순차적으로 배치가 될 것이고, offset정보를 이용해 순차적으로 데이터를 읽게 된다. 문제는 파티션4인데 컨슈머 갯수가 파티션 갯수보다 작다면 컨슈머 1,2,3중 하나가 파티션4에 접근하여 데이터를 읽게 된다. 만약 파티션 갯수와 컨슈머 갯수가 동일하개 4개씩이라면 각 컨슈머들은 하나의 파티션에서 데이터를 읽게 될것 이고, 파티션갯수가 4개고 컨슈머 갯수가 5개이면 컨슈머5는 그냥 아무일도 안하게 된다.(일반적으로 파티션 갯수와 컨슈머 갯수는 동일하게 구성하는 것을 추천한다고 함.)



![img](/assets/img/post/post-kafka/kafka-5.png)

컨슈머그룹이 존재하는 또 다른 이유가 있다. 물론 이러한 구조로 데이터를 병렬로 읽게 되어 빠른처리가 가능하다는 부분도 있겠지만, 특정 컨슈머에 문제가 생겼을 경우 다른 그룹내 컨슈머가 대신 읽을 수 있게 리벨런싱이 되어 장애 상황에서도 문제 없이 대처할 수 있게 된다.

**Broker, Zookeeper** : broker는 카프카 서버를 칭한다. 동일한 노드내에서 여러개의 broker서버를 띄울 수 있고, Zookeeper는 이러한 분산 메시지큐의 정보를 관리해주는 역할을 한다. 카프카를 띄우기 위해서는 반드시 주키퍼가 실행되어야 한다.

**Replication** : 카프카에서는 replication 수를 임의로 지정하여 topic를 만들 수 있다. replication-factor에 지정하는데 만약 3으로 하면 replication 수가 3이 된다.

Kafka Cluster에 3개의 broker가 있고 3개의 Topic이 있다고 가정해보자.

Topic-1은 replication-factor 1, Topic-2은 replication-factor 2, Topic-3은 replication-factor 3인 경우이다.

![img](/assets/img/post/post-kafka/kafka-6.png)

그렇다면 replication은 왜 필요할까? 단지 데이터의 복제 용도라기 보다는 특정 borker에 문제가 생겼을 경우 해당 broker의 역할을 다른 broker에서 즉각적으로 대신 수행 할 수 있게 하기 위한 용도 일 것이다.

#### Replication - leader & follower

replication을 좀더 자세히 들여다보면, 복제요소중 대표인 leader, 그외 요소인 follower로 나누어진다. topic으로 통하는 모든 데이터의 read/write는 오직 leader에서 이루어지고 follower는 leader와 sync를 유지함으로써 leader에 문제가 생겼을 경우 follower들 중 하나가 leader역할을 하게 되는 것이다.

![img](/assets/img/post/post-kafka/kafka-7.png)

만약 카프카 클러스터내 broker 2에서 장애가 발생되었다면, broker 2에 있던 Topic-2(leader)의 역할을 대신 수행하기 위해 아래 그림과 같이 broker 1에 있는 Topic(follower)가 leader역할을 하게 될 것이다.

![img](/assets/img/post/post-kafka/kafka-8.png)

복제된 데이터가 follower들에게 있으니, 메시지의 유실이 없다는 장점이 있지만, 복제를 하기 위한 시간과 네트워크 비용이 들기 때문에 데이터의 중요도에 따라 ack옵션으로 성능과 데이터의 중요도에 따라 다음과 같이 세부설정이 가능하다.

**ack (default:1)**

> - 0 : 프로듀서는 서버로부터 어떠한 ack도 기다리지 않음. 유실율 높으나 높은 처리량
> - 1 : 리더는 데이터를 기록, 모든 팔로워는 확인하지 않음
> - -1(또는 all) : 모든 ISR 확인. 무손실

ack값을 설정하여 데이터의 무손실에 더 중요성을 둘 것인지 또는 유실을 어느정도 감수 하더라고 속도에 중요성을 둘 것인지를 셋팅할 수 있다.



지금까지 설명한 모든 구성요소를 그림으로 표현하면 아래 그림과 같다.

![img](/assets/img/post/post-kafka/kafka-9.png)

Producer에서는 메시지를 입력하고, Consumer에서 메시지를 읽어갈때 Zookeeper에서 broker 및 offset정보를 관리하기 때문에 분산처리가 가능하게 된다.

카프카를 운영하기에 앞서 기본적인 구성요소나 매커니즘에 대해 충분히 이해를 하면 운영 하는데 많은 도움이 될 것이다.



**참고자료**

- 공식홈 문서 : <http://kafka.apache.org/documentation/>