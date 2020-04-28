# Singular Value Decomposition

> [Singular Value Decomposition Part 1: Perspectives on Linear Algebra](https://jeremykun.com/2016/04/18/singular-value-decomposition-part-1-perspectives-on-linear-algebra/)
[Singular Value Decomposition Part 2: Theorem, Proof, Algorithm](https://jeremykun.com/2016/05/16/singular-value-decomposition-part-2-theorem-proof-algorithm/)
위 두 포스트를 정리한 글
번역체가 남아있는 데 참고해주세요 :)

SVD는 CS, DA, Statistics에서 사용되는 기본적인 툴.
회귀 예측 및 optimization의 적절한 해결 찾는데도 사용됨

선형대수학이 어떻게 `subspace`와 `matrices`를 동일시 하는지 짚고 넘어가자.

주의할 점은 `transformations`으로써의 행렬과 `convenient way to organize data`로써의 행렬과의 관계성이다.

## Data vs. maps
선형대 애호가들은 `matrix factorizaiton`이라는 표현을 좋아함
그들은 아래와 같이 불투명하게(`opaque`) 말할거야!

> 실숫값을 가지는 $m \times n$ 행렬 $A$의 SVD는 $A$의 `factorization`이라 부르고 아래와 같이 쓴다.
$$U \Sigma V^T$$
$U$는 $m \times m$의 직교행렬, $V$는 $n \times n$의 직교행렬, 그리고 $\Sigma$는 대각행에 nonnegative real entries를 가지는 대각 행렬이다.

위 정의로 뭐 단어적인 의미는 알아도, 그래서 `big picture`가 뭔데?

자, 행렬에 대한 두 가지 개념으로 해석해보자.

첫 번째 해석, $A$는 $n$차원 벡터 공간에서 $m$차원 벡터 공간으로의 선형 사상이다.
- 위 해석으로 보면 `factorization`은 domain과 codomain의 basis를 바꾸는 행위이다.
- 특히, $V$는 $n$차원 벡터 공간(domain)의 basis를, $U$는 $m$차원 벡터 공간(codomain)의 basis를 바꾼다.
- `Change of Basis`: $P_{B\rightarrow B^{\prime}}$

위의 설명은 참으로 멋지다, **$A$를 구성하는 데이터가 우리가 더 알고 싶은 선형 지도에 대한 설명이라면 말이다.**
저자는 몇 년전에 Google의 `PageRank` 알고리즘을 활용하여 `linear map modeling a random walk through the internet`에 대한 분석을 진행했는데, 대부분의 선형 대수 API는 실제 행이나 열의 데이터를 포함하고 있다. 이게 두 번째 해석이다. 즉, $A$의 $\mathbb{R}^n$의 각 행(벡터)과 $m$개의 데이터가 있으며 이는 실 세계의 일부 프로세스의 **"관찰값"** 이다.
선형 대수적인 생각과 로직을 가지고 있지 않으면 우리는 위 행렬이 벡터 공간에서 다른 벡터 공간으로의 벡터 매핑이라 생각하지 않을 수 있다는 것이다!

위를 정리하면,
1. Transformations
2. Observations

으로 생각한다는 것인데, 첫 번째 관점은 사람들이 처음 선형 대수를 공부할 때 좌절감을 느끼는 부분이다. 두 번째의 관찰값, 즉 데이터 관점은 단지 표에 데이터를 넣는 것이 "말이 되는", 편리한 방법이라 생각한다.
저자의 생각으로 SVD를 접할 때 위와 같은 문제는 두 관점이 충돌하는 것이며(`perspectives collide`, 적어도 나의 경우) 인지 부조화(`cognitive dissonance`)와 함께 온다.

위 두 개의 아이디어를 결합하는 방법은 무엇일까?
Data를 linear map A의 $\mathbb{R}^n$의 basis vector들의 상(image)라 생각하는 것이다.
ㅎㅎㅎ.. 저자님 더 어려워 졌는뎁쇼...?

다행히 예시를 들어 주셨다.

![img](https://jeremykun.files.wordpress.com/2015/11/movieratings.png)

각 행은 영화에 대한 평점, 각 열은 Aisha, Bob 그리고 Chandrika가 영화에 대한 평점을 매긴 것이다. 즉, $A_{0,0}$은 Aisha가 Up에 대해 평점을 매긴게 2점이라 생각하면 되겠죠?

위에서는 1~5의 정수지만, 평점이 과연 정수로 떨어질까? 정수라는 선택지만 있으니 정수를 준 것이지 실제로는 `Real Value`, 실수라는 것을 기억하시라!
때문에 위 행렬은 `linear map`이다.

자, Domain 행 벡터들이 3차원이니 $\mathbb{R}^3$이고 Codomain은 열 벡터들이 8차원이니 $\mathbb{R}^8$이 되겠네? 각각 basis는 사람과 영화가 되겠고?

![img](https://jeremykun.files.wordpress.com/2015/11/moviemapping.png)

이러면 데이터 셋은 $A(\overrightarrow{e}_{\text{Aisha}}),A(\overrightarrow{e}_{\text{Bob}}),A(\overrightarrow{e}_{\text{Chandrika}})$로 표현된다.

codomain이 굉장히 크면 $A$의 상은 codomain의 아주 작은 차원의 선형 `subspace`가 될 것이다(왜 subspace가 되냐고? 연산이 보존되고 영점이 존재하자나!).
보면, `span`이라 되있지? 선형 대수에서 span이란 vector set의 모든 선형 결합을 모아둔 집합이다. 위에서 한 단계는 굉장히 중요한게, 우리의 관점을 단순히 개별 데이터 포인트로 보는 것이 아니라 모든 `linear combination`으로 늘렸다는 게 중요한 거야!

이게 왜 도움이 되지? (여기서 선형대수의 linear modeling assumption을 보게되짛)
만일 우리가 이 행렬을 사용하여 사람들이 영화를 어떻게 평가할 지 말하려 한다면 우리는 3 사람의 정보밖에 없기 때문에 예측할 사람을 Aisha, Bob, Chandrika의 선형 결합으로 나타낼 수 있어야 한다. 영화가 domain이어도 똑같은게 새로운 영화를 기존 영화들의 선형 결합으로 나타내야 한다.

이는 영화를 다른 영화의 선형 결합으로 표현할 수 있다! 가 아니라, **우리가 당면한 과제를 위해 어떤 추상적인 벡터 공간에서 선형 결합으로 영화를 공식으로 나타낼 수 있다는 것이다.** 다시 말해서, 우리는 영화의 등급에 추상적으로 영향을 미치는 그러한 특징들을 벡터로서 표현하고 있는 것이다. 우리는 그것을 이해할 수 있는 정당한 수학적 방법이 없기 때문에 벡터는 대체재(proxy)이다.

영화의 평정 과정이 본질적으로 "선형"이라면, 이 형식적 표현이 실제 세계를 정확하게 반영할 것이라는 희망(또는 가설을 세우거나 검증할 수 있다는 점을 제외하면, 이것이 실생활의 측면에서 무엇을 의미하는지는 완전히 불분명하다. 그것은 마치 물리학자들이 수학이 말 그대로 자연의 법칙을 지시하지 않는다는 것을 은밀히 알고 있는 것과 같다. 왜냐하면 인간은 수학을 머릿속에서 구성하고 자연을 너무 심하게 찌르면 수학은 깨지지만, 가설을 기술하는 것은 너무나(그리고 그렇게 빌어먹을 정확하고) 너무 편리해서 우리는 비행기를 설계하는 데 그것을 사용하는 것을 피할 수 없기 때문이다. 그리고 우리는 이런 목적을 위해 수학보다 더 좋은 것을 발견하지 못했다.

마찬가지로, 영화 평점은 말 그대로 linear map은 아니지만 우리는 사람들이 영화를 꽤 정확하게 어떻게 평가하는지 예측하는 알고리즘을 만들 수 있다. 그래서 스카이폴이 아이샤, 밥, 찬드리카로부터 각각 1,2,1등급을 받는다는 것을 안다면, 새로운 사람이 스카이폴을 이 세 사람과 얼마나 잘 어울리는지에 대한 선형적인 조합을 바탕으로 평가할 것이다. 즉, 선형 결합까지, 이 예에서는 아이샤, 밥, 찬드리카가 영화의 등급을 매기는 과정을 예시하고 있다.

이제 우리는 그 매트릭스를 `SVD`를 통해 사실화하는 것이 사람들이 영화를 평가하는 과정을 나타낼 수 있는 더 유용하고 대안적인 방법을 제공한다. 관련된 하나 또는 둘 다의 벡터 공간의 기초를 변경함으로써 공정의 서로 다른 (직교적) 특성을 분리한다. 우리 영화의 예에서 `"factorization"`는 다음을 의미한다.
1. 모든 영화를 어떤 vector set의 선형 결합으로 표현할 수 있는 special list vectors $v_1,v_2,\dots,v_8$를 찾아라!
2. $p_1,p_2,p_3$를 얻기 위해 위와 유사한 일을 수행하라!
3. $A$의 대각성분이 두 bases가 되도록 (1)과 (2)를 수행하라!

자, $v_i$는 그러면 `추상화된 영화`로, $p_i$를 `추상화된 사람`으로 생각할 수 있죠? 우리가 적절한 set을 찾으면 그들의 선형 결합으로 모든 영화 및 사람을 표현할 수 있으니!
위의 것이 바로 factorization의 $U$, $V$의 행, 열을 의미한다.
거듭 강조하지만(To reiterate,), 이러한 선형 결합은 영화 등급 매기기 과제에 관한 것일 뿐이다. 위에서 행렬을 대각화시키기 때문에 위에서 `special`이라 표현했다.

만약 세상이 논리적이었다면(그리고 내가 그렇다고 말하는 것은 아니다) $v_1$은 "액션 영화"라는 어떤 이상화된 개념과 일치하고, $p_1$은 "액션 영화 애호가"라는 어떤 이상화된 개념과 일치할 것이다. 그렇다면 왜 이런 기초에서 매핑이 대각선이 되는지 이해가 된다: 액션 영화 애호가들은 액션 영화만 좋아하기 때문에 $p_1$은 $v_1$을 제외한 모든 것에 0의 등급을 부여한다. 영화는 어떻게 (선형적으로) '이상화'된 영화로 분해되는가에 의해 대표된다. 임의의 숫자를 보충하기 위해, 스카이폴은 액션 영화 2/3, 디스토피아 공상과학 영화 1/5, 그리고 -6/7 코미디언 로맨스일지도 모른다. 마찬가지로 사람은 (선형적인 조합을 통해) 액션 영화 애호가, 롬-컴 애호가 등으로 분해되는 방식으로 대표될 것이다.

분명히 하자면, `Singular Value Decomposition`은 이상적인 공상 과학 영화를 발견하지 못한다. 단수값 분해의 "이상적"은 선형성의 가정과 결합된 데이터의 고유한 기하학적 구조에 관한 것이다. 이것이 인간이 영화를 분류하는 방법과 전혀 관련이 있는지는 별개의 질문이고, 대답은 No다.

1. All people rate movies via the same linear map.
2. Every person can be expressed (for the sole purpose of movie ratings) as linear combinations of “ideal” people. Likewise for movies.
3. The “idealized” movies and people can be expressed as linear combinations of the movies/people in our particular data set.
4. There are no errors in the ratings.

저자는 추가로 위와 같이 데이터 셋을 제공하면 선형대로 실 세계를 표현할 수 있다! 라고 주장하며 철학적, 윤리적 및 문화적인 얘기로 들어갈 수도 있지만 알고리즘으로 눈을 돌리자고 말한다.

뭐, 나도 그런 철학적이거나 한 목적으로 보는 글이 아니기에....

## Approximating Subspaces
SVD는 새로운 basis로 완벽히 매핑하는 것이 아닌, 매핑을 `근사`하려 사용하는 것.
앞으로 `low-dimensional linear things`는 아래 내용을 뜻한다.
```
행렬 A가 주어졌을 때, 아래와 같은 조건의 행렬 B를 찾는 것.
1. A와 측도론적으로 유사한 행렬 (measurably)
2. A에 비해 낮은 rank를 가지는 행렬
https://en.wikipedia.org/wiki/Low-rank_approximation
```

(아래 무슨 말인지 이해가 ;;)
$A$가 low rank가 아니라는 것을 어떻게 알 수 있을까?
아주 조금이라도 noise가 섞인 데이터는 압도적인 확률로 full rank이다.
- full rank? 해당 행렬이 가질 수 있는 최대로 가능한 rank의
- https://twlab.tistory.com/22
- free-variable이 없다.

low rank 행렬 공간은 행렬 내부 공간에 작은 차원을 가진다(manifold 가설).
때문에 단일 entry를 perturb시키는 것만으로 rank를 증가시킬 수 있다.(으...음>?)

SVD를 이해하기 위해 manifold를 이해할 필요는 없다.
위 people-movie 예시에서 full-rank 성질은 명백하다.
인간의 noise, randomness, arbitrariness는 완벽한 선형 결합을 불가능하게 만들고 이는 우리가 바라는 희망밖에 되지 않을 것이다.
이는 A의 이미지가 codomain의 large-dimensional subspace임을 의미한다.

low-rank approximation을 찾는 것은 데이터의 노이즈를 `smoothing`하는 것이라 생각할 수 있다.
이는 linear (이해하고 써야겠다...)

뭐 위에 글이 길게 있는데 결국 설명하고 싶은 바는 아래다.

```
SVD는 data에 맞는 line을 찾는 greedy optimization 문제를 반복적으로 푸는 것이다.
```

나는 글은 진짜 안맞아... 수식 및 알고리즘으로 보자!!

---

자, best-approximating k-dimensional linear subspace를 찾는 것부터 시작하자.
