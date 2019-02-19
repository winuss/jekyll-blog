---
layout: post
comments: true
title: '[Angular] Observable'
categories: [dev]
tags: [angular]
date: 2019-02-17
---
옵저버블은 ES7 릴리스에 포함될 비동기 데이터를 관리하기위한 새로운 표준이다. Angular는 이벤트 시스템과 HTTP 서비스에서 옵저버블을 광범위하게 사용한다.

![Observable](/assets/img/post/angular-observable/observable.png){: width="600px" }

### Observables
옵저버블은 시간이 지남에 따라 여러값을 가질 수 있는 지연 콜렉션이다.

**1. Observables는 lazy하다.**

지연 옵저버블을 뉴스 레터로 생각할 수 있다. 각 구독자(subscriber)마다 새로운 뉴스 레터가 만들어진다. 그 뉴스레터들은 구독자들에게만 보내고 다른 사람에게는 보내지 않는다.

**2. Observables는 시간이 지남에 따라 여러 값을 가질 수 있다.**

뉴스 레터 구독을 계속 열어두면, 매번 새로운 뉴스 레터를 받게된다. 발신자(sender)는 받은 시간을 결정하지만 받은 편지함에 곧바로 올 때까지 기다려야한다.

옵저버블과 프로미스간의 다른 중요한 차이점이 있는데, promise은 항상 오직 하나의 값만을 반환한다는 점이다. 또 하나는 옵저버블의 구독을 취소 할 수 있다는 점이다. 뉴스 레터를 더 이상 원하지 않으면 구독을 취소하면된다. 반면 프로미스는 취소 할 수 없다. 프라미스가 당신에게 건네지면, 그 프라미스의 resolve가 이미 진행되고 있으며, 일반적으로 프라미스의 resolve가 실행되는 것을 막을 수 있는 권한이 없다.

**Push vs pull**

observables를 사용할 때 이해해야 할 핵심 사항은 observables가 push한다는 것이다. (옵저버블은 push 시나리오를 따른다는 말) push와 pull은 데이터 생성자가 데이터 소비자와 커뮤니케이션하는 방법을 설명하는 두 가지 방식이다.

**Pull**

Pulling일 때 데이터 소비자는 데이터 생성자로부터 데이터를 가져 오는 시점을 결정한다. 생산자는 언제 데이터가 소비자에게 전달되는지를 알지 못한다.
모든 자바 스크립트 함수는 pull 시나리오를 사용한다. 함수는 데이터의 프로듀서이며 함수를 호출하는 코드는, 호출에서 하나의 반환 값을 “꺼내”(pull) 가져와 이를 소비한다.

**Push**

Pushing일 때, 다른 방향으로 동작한다. 데이터 생성자 (뉴스 레터 생성자)는 소비자 (뉴스 레터 구독자)가 데이터를 가져 오는 시점을 결정합니다.

프로미스는 오늘날 자바 스크립트에서 사용하는 가장 일반적인 push 방법입니다. 프로미스(생산자) 전달자는 등록된 콜백(소비자)에게 resolve된 값을 전달하고, 프로미스는 함수와는 달리 콜백에 그 값이 “푸시 (push)”되는 시기를 정확하게 결정한다.

Observables는 JavaScript로 데이터를 푸시하는 새로운 방법이다. 옵저버블은 여러 값의 생산자로서 구독자에게 “푸시 (pushing)”한다.

### Angular에서 Observables
Angular에서 HTTP 요청을 하면 옵저버블 형태로 반환을 한다.

~~~typescript
import { Observable } from "rxjs/Rx"
import { Injectable } from "@angular/core"
import { Http, Response } from "@angular/http"

@Injectable()
export class HttpClient {

    constructor(
        public http: Http
    ) {}

    public fetchUsers() {
        return this.http.get("/api/users")
            .map((res: Response) => res.json())
    }
}
~~~

observable을 반환하는 fetchUsers 메서드를 사용하여 간단한 HttpClient를 만들었다. 어떤 종류의 리스트에 사용자를 표시하고 싶으므로 fetchUsers 메서드를 사용해서 해보자. 이 메소드는 옵저버블을 반환하기 때문에 우리는 그것을 구독해야 한다. Angular에서 우리는 두 가지 방식으로 Observable을 구독 할 수 있다.

**방식 1**

비동기 파이프를 사용하여 템플릿의 옵저버블을 구독하는 방법이 있다. 이로 인해 Angular는 컴포넌트의 생명주기 동안 구독을 처리한다. Angular는 자동으로 구독하고 구독취소한다. 비동기 파이프가 노출되야하므로 모듈에 “CommonModule”을 import하자.

~~~typescript
import { Component } from "@angular/core"
import { Observable } from "rxjs/Rx"

// client
import { HttpClient } from "../services/client"

// interface
import { IUser } from "../services/interfaces"

@Component({
    selector: "user-list",
    templateUrl:  "./template.html",
})
export class UserList {

    public users$: Observable<IUser[]>

    constructor(
        public client: HttpClient,
    ) {}

    // 컴포넌트 초기화시 사용자들을 불러옴
    // fetchUsers 메소드는 관찰 가능 객체를 반환
    public ngOnInit() {
        this.users$ = this.client.fetchUsers()
    }
}
~~~

~~~html
<!-- 비동기 파이프를 사용하여 자동으로 구독/구독취소 함 -->
<ul class="user__list" *ngIf="(users$ | async).length">
    <li class="user" *ngFor="let user of users$ | async">
        {{ user.name }} - {{ user.birth_date }}
    </li>
</ul>
~~~
> 일반적으로 달러($)기호는 Stream의 약어로 쓰이기 때문에 변수이름 앞에 $를 사용하면 변수의 관찰 가능 여부를 쉽게 식별 할 수 있다.

**방식 2**

`subscribe()` 메소드를 사용하여 옵저버블을 구독한다. 데이터를 표시하기 전에 먼저 데이터에 뭔가 작업하기를 원한다면 편리할 수 ​​있다. 단점은 구독을 직접 관리해야한다는 것다.(구독취소를 해주어야 함)

~~~typescript
import { Component } from "@angular/core"

// client
import { HttpClient } from "../services/client"

// interface
import { IUser } from "../services/interfaces"

@Component({
    selector: "user-list",
    templateUrl:  "./template.html",
})
export class UserList {

    public users: IUser[]

    constructor(
        public client: HttpClient,
    ) {}

    // 수동으로 구독하고 사용자를 가저옴
    public ngOnInit() {
        this.client.fetchUsers().subscribe((users: IUser[]) => {
            // do stuff with our data here.
            // ....
            // asign data to our class property in the end
            // so it will be available to our template
            this.users = users
        })
    }
}
~~~

~~~html
<ul class="user__list" *ngIf="users.length">
    <li class="user" *ngFor="let user of users">
        {{ user.name }} - {{ user.birth_date }}
    </li>
</ul>
~~~

템플릿 로직이 꽤 비슷하다는 것을 알 수 있듯이, 2번 방식으로 간다면 구성 요소 논리는 훨씬 더 복잡해 질 수 있다. 일반적으로 방식1을 선택하는 것이 좋다. 가장 쉽고 구독을 수동으로 관리할 필요가 없다. `2번방법에서 구독을 사용하지 않는 동안 열어두면 메모리 누수가 발생하므로 좋지 않다.`

### Creating an observable yourself

Angular가 제공한 일반적인 옵저버블을 다루는 방법을 알았으므로 옵저버블을 어떻게 생성하는지 보자.

~~~typescript
import { Observable } from "rxjs/Observable"

// create observable
const simpleObservable = new Observable((observer) => {
    
    // observable execution
    observer.next("bla bla bla")
    observer.complete()
})

// subscribe to the observable
simpleObservable.subscribe()

// dispose the observable
simpleObservable.unsubscribe()
~~~

observables는 새로운 Observable() 호출을 사용하여 만든 다음 observer에 가입하고 next()를 호출하여 실행하고 unsubscribe()를 호출하여 삭제한다.

**observable 만들기**

observables를 만드는 것은 쉽다. 새로운 Observable()을 호출하고 옵저버를 나타내는 하나의 인수를 전달하면됩니다. 그러므로 저는 보통 그것을 “observer”라고 부른다.

**옵저버블 구독하기**

옵저버블은 구독하지 않으면 아무 일도 일어나지 않을 것이다. 옵저버를 구독 할 때 subscribe()를 호출 할 때마다 옵저버블에 독립 설정이 실행된다. 구독 요청은 동일한 옵저버블에 대한 여러 구독자간에 공유되지 않는다.

**옵저버블 실행하기**
observables 안의 코드는 observables의 실행을 나타낸다. 옵저버블을 만들 때 주어진 매개 변수(observer)에는 옵저버블의 구독자에게 데이터를 보낼 수있는 세 가지 함수가 있다.

- next: Number나 Array나 객체같은 여러 값을 subscribers에게 보낸다.
- error: 자바스크립트 에러나 예외값을 보낸다.
- complete : 어떤 값도 보내지 않는다.

`next` **콜은 구독자에게 실제로 데이터를 전달할 때 가장 일반적이다.** 옵저버블의 실행 중에는 `observer.next()`의 무한 호출이있을 수 있지만 `observer.error()` 또는 `observer.complete()`가 호출되면 실행이 중지되고 더 이상 데이터가 subscribers에게 전달되지 않는다.

**옵저버블 처분**
옵저버블의실행은 무한한 시간 동안 실행될 수 있기 때문에, 실행을 막을 수있는 방법이 필요하다. 각 구독자마다 각 실행이 실행되기 때문에, 메모리와 컴퓨팅 성능이 낭비, 즉 더 이상 데이터가 필요없는 구독자는 구독을 멈추는 것이 중요하다.

옵저버블을 구독할 때, 진행중인 실행을 취소하면 구독이 반환된다. 실행을 취소하려면 `unsubscribe()`를 호출하면 된다. 