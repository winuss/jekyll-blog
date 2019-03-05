---
layout: post
comments: true
title: "OpenSSH Server를 WSL(Windows Subsystem for Linux)에서 구동하기"
categories: [dev]
tags: [tip]
date: 2019-03-05
---
![headerimg](/assets/img/post/openssh-server-wsl/linux-ubuntu-windows-10.jpg)
WSL(Windows Subsystem for Linux)에서 OpenSSH Server를 띄우고 접속하는 방법에 대해 알아보겠다.

# sshd_config

먼저 Windows10에서 Windows Subsystem for Linux(WSL)을 설치한 후 CMD화면에서 bash명령으로 WSL에 진입한다.

`/etc/ssh/sshd_config`를 수정해주자.

~~~bash
$ sudo vi /etc/ssh/sshd_config
~~~
> `Port 8889` # ssh 접속 포트를 기입한다. 22는 윈도우 자체 내장 SSH서버가 이번호를 선점하고 있기 때문에 반드시 피해야 한다.

> `UsePrivilegeSeparation no `# SWL이 chroot()를 지원하지 않기 때문이라고 하는데 이 부분은 패스 했다. (안해도 되는듯..)

> `PasswordAuthentication no` # SSH Key를 사용할 것이기 때문에 no로 설정

~~~bash
$ sudo service ssh restart
~~~

정상적으로 서버가 구동 되었다면 윈도우 방화벽에서 인바운드에  설정한 포트번호를 허용해주자.

# SSH Key인증을 이용한 접속 (Password 없이 접속)

위에서 설정한 서버에 클라이언트가 접속하기 위해 키를 생성하여 공개키 정보를 서버에 등록해주면 클라이언트에서는 별도의 패스워드 입력없이 접속할수 있게 된다.

먼저 키파일을 생성하자.

~~~bash
$ ssh-keygen -t rsa
~~~

`~/.ssh` 디렉토리에 id_rsa, id_rsa.pub, known_hosts 파일들이 생성된다.

자, 아래 명령으로 공개키 파일을 서버로 전송하자. 수동으로 복사해도 상관없지만 `ssh-copy-id`를 사용해 쉽게 전송하자.

~~~bash
$ ssh-copy-id -i ~/.ssh/id_rsa.pub [user]@[host] -p 8889
~~~

>`Permission denied (publickey)` 

권한 오류가 발생한다..

권한 오류가 발생하는 원인은 우리가 서버에서 설정한 `/etc/ssh/sshd_config`에 `PasswordAuthentication`를 `no`로 설정했기 때문이다. 아직 key기반의 접속이 셋팅이 안된 상태이기 때문에 패스워드를 입력할 수 있도록 잠시 `PasswordAuthentication yes`로 변경한 후 다시 시도 해보자.

~~~bash
$ ssh-copy-id -i ~/.ssh/id_rsa.pub [user]@[host] -p 8889
~~~

>Now try logging into the machine, with:   "ssh '[user]@[host]'"<br>
>and check to make sure that only the key(s) you wanted were added.

복사가 정상적으로 이루어 졌다면 해당 서버의 `PasswordAuthentication`를 `no`로 다시 변경하자.(구지 다시 no로 변경할 필요는 없다. key기반의 접속이기때문에 자동으로 패스워드를 확인하는 절차는 패스되는 것 같다.)