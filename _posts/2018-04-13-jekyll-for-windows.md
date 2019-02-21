---
layout: post
comments: true
title: "Jekyll을 Windows(윈도우 10)에 설치하기"
categories: [dev]
tags: [tip]
date: 2018-12-13
---
![headerimg](/assets/img/subcate/jekyll-head.png)
Jekyll은 기본적으로 윈도우는 지원하지 않는다.
윈도우에서 지킬을 설치하기 위한 가장 간편한 방법은 Windows Subsystem for Linux를 이용하는 것이다.

## 윈도우에서 지킬 설치하기
{:.no_toc}
0. list
{:toc}
먼저 Windows10에서 Windows Subsystem for Linux(WSL)을 설치한다.

CMD > Bash - WSL 진입

## 설치 스크립트
{:toc}
**1.패키지 업데이트**
~~~bash
$ sudo apt-get update -y && sudo apt-get upgrade -y
~~~

**2.ruby 설치**
~~~bash
$ sudo apt-get install make build-essential
$ sudo apt-get install ruby ruby-dev
~~~

설치과 완료되는 환경변수에 등록해주자.

**환경변수 등록**
~~~bash
$ echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
$ echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
$ echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
$ source ~/.bashrc
~~~

**3.지킬 설치**
~~~bash
$ gem update
$ gem install bundler
$ gem install jekyll

$ gem install rake && bundle install
~~~

**4.지킬 버전확인**
~~~bash
jekyll -v
~~~

## 새로운 사이트 생성 및 실행
{:toc}
~~~bash
mkdir new myblog
cd myblog
~~~

**사이트를 빌드하고 로컬 서버에서 사용할 수 있도록 한다.**
~~~bash
jekyll serve
~~~

**NOTE**: http://127.0.0.1:4000 에 접속하면 서버가 생성 되었다. 접속 GO!

참고 : [jekyll 공식홈페이지](https://jekyllrb.com/docs/installation/windows/)