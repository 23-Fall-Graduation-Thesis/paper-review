# Robust Fine-Tuning of Deep Neural Networks with Hessian-based Generalization Guarantees

#### Link

https://arxiv.org/abs/2206.02659

#### Information

- Author/Institution : Haotian Ju, Dongyue Li, Hongyang R. Zhang
- Conference/Journal : Proceedings of Machine Learning Research (PMLR), 2022
- Cited by 10 (2023.09.12)
- Submitted on 6 Jun 2022 (v1), last revised 7 Aug 2023

### Abstract

우리는 fine-tuning의 일반화와 관련된 특징에 대하여 연구한다.
현재는 초기값(pre-trained)에서부터 fine-tuning된 모델간의 distance, 그리고 noise stability property를 이용하여 지금 모델이 얼마나 일반화 성능이 뛰어난지를 가늠하기 위해 사용한다.

<br>

## Introduction

Fine-tuning은 pretrained 모델을 실제 문데에 사용하기 위해 쓰이는 일반적인 접근법이다. 이로 인해 supervised learning을 여러가지 활용이 가능해졌지만, 그와 동시에 fine-tuning 과정에서 overfitting이 발생하는데 이 overfitting문제를 이해하는 것이 바로 challenging한 문제이다.

선행 연구를 기반, distance-based regularization을 함으로서 overfitting 문제를 해결할 수 있음이 입증되어있다.

- 초기 가중치로부터의 거리가 일반화에 결정적으로 영향을 미친다.
- 거리 기반의 정규화를 사용하여 과적합을 완화할 수 있다.

본 연구에서는 Hessian을 기반으로 distance measure를 측정하여 보다 정교한 generalization bound를 계산하는것을 목적으로 한다.

- PAC-Bayesian analysis approach : [2], [33]
- Deep net Hessian : [54]

> [!abstract]
> In this work, we analyze the generalization error of fine-tuned deep models using PAC-Bayes analysis[^1] and data-dependent measurements based on deep net Hessian[^2]. With this analysis, we also study the robustness of fine-tuning against label noise[^3].

<br>

### bound

We prove that on a training dataset of size $n$ sampled from a distribution $D$, with high probability, the generalization error of a fine-tuned $L$-layer network $f_W$ is bounded by:

$$
\sum^L_{i=1} \sqrt{\dfrac{\max_{(x,y)\sim\mathcal D}v_i^\top \mathbf{H}_i^+[l(f_W(x),y)]v_i}{n}}
$$

-> Theorem 2.1에서 상세한 유도 설명이 있음
-> Theorem 2.1에서 구한 관계에서 우변의 loss를 좌변으로 이항하여 좌변을 generalization error 형태로 표현한 뒤, 상수값을 소거하면 위의 공식을 구할 수 있다.

$v_i$에 대한 norm으로 constraint가 존재할 때, distance based regularization of fine-tuning에 대한 bound를 계산할 수 잇다.

Distance based measure를 이용하여 generalization error를 추정하는 것 보다 Hessian Distance measure를 이용하여 측정할 때 모델의 실제 경험적인 generalization error를 보다 더 정확하게 추정하는 것을 확인할 수 있다.
![figure1|600](https://i.imgur.com/xDuuRLY.png)

<br>

### Related work

<br>

## Hessian-based generalization

### Problem Setup

#### Notation

Consider predicting a target task given a training dataset of size $n$.

- **Dataset**
  - $(x_i, y_i)$ for $i=1, \cdots, n$
    - $x_i$ : feature vector $\in \mathbb R^d$
    - $y_i$ : class label between $1$ to $k$.
  - Assume the training examples are drawn independently from an unknown distribution $\mathcal D$
  - $\mathcal X$ : support set of the feature vectors of $\mathcal D$
- **FeedForwardNeuralNetwork : FFNN**
  - $f_W$ : $L$ layer feed-forward neural net
    - with initialized with weight metrix $\hat{W_i}^{(s)}$ for $i \in \{1, \cdots, L\}$
    - dimension of layer $i$, $W_i$ is $d_i$ by $d_{i-1}$
    - $d_i$ : layer $i$의 output dimension
      $\rightarrow$ 따라서 $d_0$은 input dimension $d$를 의미하며 $d_L$은 output dimension $k$을 의미한다.
    - $\phi_i(\cdot)$ : layer $i$의 activation function.
  - Given a feature vector $x \in \mathcal X$,the output of $f_W$ is $$f_W(x) = \phi_L\Big(W_L \cdot \phi_{L-1}\Big(W_{L-1}\cdots \phi_1\big(W_1 \cdot x\big)\Big)\Big)$$
- **loss funuction**
  - $\ell :\mathbb R^k \times \{1, \cdots, k\} \rightarrow \mathbb R$ : loss functio
  - $\ell (f_W(x), y)$ : loss of $f_W$
  - $\hat {\mathcal L}(f_W)$ : empirical loss = training examples에 대한 $f_W$의 averaged loss  
    _실제로 우리가 학습에 사용하는 sample data에 대해서 경험적으로 구할 수 있는 loss_
  - $\mathcal L(f_W)$ : expected loss of $f_W$ = expectation of $\ell (f_W(x), y)$ over $x$ sampled from $\mathcal D$ with label $y$  
    _loss function을 실제로 게산하여, loss function에 대한 expected value를 구한 것_ (data의 distribution $\mathcal D$에서 각각의 데이터가 샘플링될 "확률"에 기반하여 기댓값을 계산) $$\mathcal L(f_W) = \int_{X\times Y} \ell (f_W(\vec x), y) p_{X,Y}(\vec x,y) d\vec xdy $$
- **generalization error** ([wiki: generalization error](https://en.wikipedia.org/wiki/Generalization_error))
  defined as its expected loss minus its emplircal loss $$ \mathcal L(f*W) - \hat{\mathcal L}(f_W)$$
  \_An Algorithm is said to generalize if :* $$\lim_{n\rightarrow \infty} \mathcal L(f_W) - \hat{\mathcal L}(f_W) = 0$$

- Notations...

  - $\|v\|$ : Euclidean norm of $v$, for any vector $v$.
  - $\|X\|_F$ : Frobenius norm[^9] of matrix $X \in \mathbb R^{m\times n}$
  - $\|X\|_2$ : spectral norm of matrix $X$
  - $\|X\|_{1, \infty}$ be defined as $\max_{1\le j\le n}\sum^m_{i=1} |X_{i, j}|$
  - $\text{Tr}[X]$ : trace[^4] of $X$ (if $X$ is a squared matrix.)
  - $\langle X, Y\rangle = \text{Tr}[X^\top Y]$ : matrix inner product[^5] of $X$ and $Y$.
  - $\mathbf H_i[\ell(f_W(x), y)]$ : weight matrix $W_i$에 대한 loss function의 Hessian matrix ($d_id_{i-1})$  
    (= $\mathbf H_i$ : weight matrix $W_i$에 대하여 loss function의 Hessian)
  - $\mathbf H_i^+[\ell(f_W(x), y)] = U\max(D,0)U^\top$ : Hessian atrix의 eigen-decomposition $UDU^\top$에 대하여 positive eigen-space까지만 취하여 다시 복호화한 행렬
    (=$\mathbf H_i^+$ : Hessian matrix에서 non-negative eigenvalue를 가지는 부분까지만 취한 행렬)
  - $v_i$ : flatten vector of $W_i - \hat{W}_i^{(s)}$ for every $i = 1, \cdots, L$  
    _기존에 있던 weight랑 fine tuning 완료했을 때의 weight 행렬의 차이에 대해서 계산한 행렬을 벡터로 flatten한 것_

- for two functions $f(n)$ and $g(n)$, we write $g(n) = O(f(n))$ if there exists a fixed value $C$ that does not grow with $n$ such that $g(n) \le C \cdot f(n)$ when $n$ is large enough. (시간복잡도; 최악의 정의)

<br>

### Generalization bounds using PAC-Bayesian

##### Concept

fine-tunning의 performance를 더 정확하게 이해하기 위하여, generalization bound를 분석하는데 중요하다고 알려져있는 PAC-Bayesian approach를 사용한다. (선행연구 _Arora et al [2]; Stronger generalization bounds for deep nets via a compression approach”_ 참고)

- PAC-Bayesian에서는 모델의 weight에 가한 pertubation이 모델의 예측성능에 얼마나 영향을 주는지에 따라 모델의 일반화 성능이 결정된다고 한다.
- deep network의 stronger generalization bound는 네트워크가 가지고 있는 noise stability property에 따라 결정된다.
- 다양한 레이어에 걸친 activation function의 **Lipschitz-연속성**[^6]과 smothness을 사용하여 network의 error resilience; 오류 복원력을 어떻게 정량화할 것인지를 선행연구에서 진행했다.
- 그러나, 여전히 선행연구에서 계산한 generalization bound는 실제 empriizal generalization error와 큰 gap이 존재한다.
  => 어떻게 더 실제 경험적인 일반화 성능을 잘 포착할 수 있는 방법이 없을까?

> [!important]
> The key idea of our approach is to measure generalization using the Hessian of $\ell(f_W(x), y)$

##### Formulation

- prior distribution $\mathcal P$ : preitrained weights matrix $\hat{W}^{(s)}$에 대한 noisy pertubation의 분포
- posterior distribution $\mathcal Q$ : fine-tuned weights matrix $\hat{W}$에 대한 noisy pertubation의 분포
- $\ell_{\mathcal Q} (f_W(x), y)$ : distribution $\mathcal Q$를 따르는 noisy perturbation을 가한 모델 $f_W(x)$의 loss
- $\mathcal I(f_W(x), y) =\ell_{\mathcal Q} (f_W(x), y) - \ell (f_W(x), y)$
  - lower value of $\mathcal I(f_W(x), y))$ means the network is more error resilient.
  - 이 값이 더 적을수록, perturbation을 가했을 때와 가하지 않았을 때의 loss의 차이값이 작다는 의미이기 때문에 네트워크가 더 회복력 있다/탄력있다 -> 강건하다라고 말할 수 있을 것이다.

> [!important]
> To illustrate the connection between Hessian and generalization, consider a **Taylor’s expansion**[^7] of $\ell(f_{W+U}(x),y)$ at $W$, for some small perturbation $U$ over $W$

weight $W$에 대한 small perturbation $U$가 더해져 있을 때의 loss에 대한 테일러 전개를 사용하여 Hessian과 generalization간의 관계를 분석한다.

perturbation $U$ 에 대하여, mean of $U$와 covariance matrix를 $\Sigma$ 라고 가정하면, $\mathcal I(f_W(x), y)$를 <span style='color:#eb3b5a'>Hessian과 perturbation으로 표현할 수 있다.</span>

$$\mathcal I(f_W(x), y) = \langle \Sigma, \mathbf H[\ell(f_W(x), y]\rangle + H.O.T$$

- [*] _Analysis_

  - 좌변 : weight에 perturbation을 준 모델의 loss와 기존 모델의 loss의 차이
  - 우변 : perturbation의 covariance matrix와 기존 모델의 loss의 Hessian의 inner product(=행렬곱의 trace)에 테일러 전개의 higher-order-term.

  즉, 직관적으로 loss의 차이가 작을수록 모델의 일반화가 잘 되어있다는 의미이다(좌변). 마찬가지로 우변의 값은 작을수록 perturbation matrix의 분포가 loss Hessian의 분포와의 차이가 크다는 의미이기 때문에 직관적으로 유사하게 해석할 수는 있다.

위의 전개는 모든 sample에 대하여 적용할수는 있지만, uniform convergence[^8]를 하는지에 대해서는 확실하지 않다.

이는 모든 신경망 레이어에서의 perturbation에 대한 헤시안$\mathbf H_i\ \ \forall i = 1, \cdots, L$ 분석을 필요로 하지만, activation function 도함수의 Lipschitz-연속성과 부드러움을 가정하면 얻을 수 있다.

#### Theorem 2.1

Assume the activation functions $\phi_i(·)$ for all $i = 1,...,L$ and the loss function $\ell(·,·)$ are all twice differentiable, and their first-order and second-order derivatives are all Lipschitz-continuous.
Suppose $\ell(x,y)$ is bounded by a fixed value $C$ for any $x \in \mathcal X$ with class label $y$. Given an $L$-layer network $f_{\hat W}$ , with probability at least 0.99, for any fixed $\epsilon$ close to zero, we have $$ \mathcal L(f*{\hat{W}}) \le (1+\epsilon) \hat {\mathcal L}(f*{\hat W}) + \frac{(1+\epsilon)\sqrt{C}\sum^{L}_{i=1}\sqrt{\mathcal H_i}}{\sqrt n} + \xi$$
where $\mathcal H_i$ is defined as $\max_{(x,y)\in \mathcal D} v*i^\top \mathbf H^+\_i [\ell(f*{\hat W} (x),y)]v_i$, for all $i = 1,...,L,$ and $\xi = O(n^{−3/4})$ represents an error term from the Taylor’s expansion. [[#Notation]]

- [*] _Analysis_
  - twice-differentiable : 2차 미분가능
  - first-order / second-order derivatives are all Lipschitz-continuous : 1차, 2차도함수 모두 Lipschitz 연속성을 가정
  - loss function $\ell(x,y)$가 dataset에 존재하는 어떤 sample $(x,y)$에 대해서도 fixed value $C$에 대해 bound된다고 가정하자
  - <span style='color:#0fb9b1'>with probability at least</span> $0.99$(?) : 공식이 거의 확실하게 만족된다.
  - <span style='color:#0fb9b1'>for any fixed</span> $\epsilon$ <span style='color:#0fb9b1'>close to zero</span>(?) : 0에 가까운 어떤 작은상수... epsilon-delta 논법 다룰때 느낌
  - $\mathcal H_i$ : dataset에 존재하는 어떤 $(x,y)$에 대하여, 어떤 layer $i$에 대한 loss의 Hessian matrix에서 positive eigenvalue만 취해서 만든 행렬의 앞뒤에 layer간의 weight의 차이를 flatten한 벡터를 곱하여 얻는 스칼라값
    - $\mathbf H^+$는 eigenvalue가 양수인 곳까지만 취한 행렬이기 때문에 eigen value가 모두 양수라서 positive definiite이기 떄문에 위 연산의 결과는 항상 양수인 스칼라값이 될 것이다.
    - 따라서 이차 형식이 최소값 0을 가지며, 이 값은 오직 벡터 $v_i$가 영 벡터일 때만 달성된다.
  - $\xi$는 테일러 전개로서 $n^{-3/4}$라는 다항함수로 bound된다.
    이런 항들을 이용하면...expected loss는 empirical loss에 대하여 위와 같은 upper bound를 가진다. 반대로 이야가히면 expected loss를 empirical loss에 대한 하한으로 표현할 수 있다.

Theorem 2.1은 early stopping을 허가너 하지않는 vanilla fine-tuning에 적용할 수 있다. 또한 위의 결과를 이용하면 기존에 사용되는 distance-based regularization에 그대로 적용할 수 있다.

##### distance-based Regularization

$$\left\|W_i - \hat{W_i}^{(s)}\right\|_F \le \alpha_i , \ \forall i = 1, \cdots, L$$
distance-based regularization은 위와 같이, pretrained 모델에서 가져온 initial weight와 학습된 뒤의 weight의 차이값에 대한 norm을 모든 layer마다 특정값이 넘지 않게 강하게 제약을 두는 것이다.

위에서 우리가 Theorem 2.1에서 유도한 공식을 활용하면 $\mathcal{H_i} \le \alpha^2(\max_{(x,y)\in \mathcal X}  \text{Tr}[\mathbf{H_i^+}(\ell(f_{\hat{W}}(x), y))])$ 으로 표현할 수 있기 때문에, distance-based regularization처럼 사용할 수 있다.

### Incorporating consistent losses with distance-based regularization

fine-tuning model의 noisy label에 대한 robustness를 분석한다.
We consider a random classification noise model

- each label $y_i$ is independently flipped to $1, \cdots, k$ with some probability
- noisy label $\tilde{y}$
- conditional label noise를 적용하면, $\tilde{y}$는 $z$의 $k\times k$ confusion matrix $F_{y,z}$ 에 대한 확률과 동일하다.

선행연구에서는 noisy label을 학습할 때 사용되는 statistically-consistent loss[^10]를 minimizing하는 것을 제안한다.

(labeling noise... 생략)

<br>

### Proof overview

(지금은 생략)

먼저, 신경망의 noise stability가 layer-wise Hessian approximation으로 표현할 수 있음을 보인다.

$U \sim \mathcal Q$는 posterior distribution $Q$로부터 추출한 random variable일 때, perturbed loss는 $\ell_{\mathcal Q}(f_U(x), y)$ 로 표현되는데, 각 $U$에 대한 loss의 expectation을 의미한다.

#### Lemma 2.3

In the setting of [[#Theorem 2.1]], for any $i = 1,2,··· ,L$, let $U_i \in \mathbb{R}^{d_id_{i−1}}$ be a random vector sampled from a Gaussian distribution with mean zero and variance $Σ_i$. Let the posterior distribution $\mathcal Q$ be centered at $W_i$ and perturbed with an appropriately reshaped $U_i$ at every layer.  
Then, there exists a fixed value $C_1 > 0$ that does not grow with $n$, such that the following holds for any $x \in \mathcal X$ and $y \in \{1,...,k\}$
$$ \ell*{\mathcal Q}(f_W(x), y) - \ell(f_W(x), y) \le \sum^L*{i=1} \left(\langle\Sigma_i, \mathbf{H}\_i[\ell(f_W(x), y)] \rangle+C_1\|\Sigma_i\|\_F^{3/2}\right)$$
Lemma 2.3에서 얻은 Hessian approximation에 PAC-Bayes을 적용하여 bound를 구할 수 있다.

KL-divergence[^11] between the prior distribution and the posterior dietribution is equl to (?)
$$\sum^L_{i=1}\Big\langle\Sigma^{-1}_i, v_iv_i^\top \Big\rangle$$
Lemma 2.3의 우변에서 각 layer에 대한 KL-divergence값을 대입하고 $\mathbf{H}^+$를 이용하면,
minimizing the sum of Hessian estimate ane the above KL-divergence in the PAC-Bayes bound will lead to a different covariance matrix for every later, which depends on $v_i$ and $\mathbb{E}[\mathbf{H}^+_i]$
$$\sum^L_{i=1} \left(\langle\Sigma_i, \mathbb{E}_{x,y}[\mathbf{H}_i[\ell(f_W(x), y)]] \rangle+ \frac{1}{n}\langle\Sigma^{-1}_i, v_iv_i^\top \rangle\right) \le \sum^L_{i=1} \left(\langle\Sigma_i, \mathbb{E}_{x,y}[\mathbf{H}^+_i[\ell(f_W(x), y)]] \rangle+ \frac{1}{n}\langle\Sigma^{-1}_i, v_iv_i^\top \rangle\right) $$

따라서, 위의 공식을 공분산 행렬들에 대해 최소화하면, Theorem 2.1에서 보인 generalization bound를 얻을 수 있다. 공분산 행렬에 대한 최소화는 deterministic한 해를 얻을 수 있고, training data의 무작위성에 따라 변하지 않도록 고정되기 때문에 어떤 task에도 적용할 수 있다.

#### Lemma 2.4

(생략; 라벨노이즈)

## Exmperiment

We experiment with seven methods, including fine-tuning with and without early stopping, distance-based regularization, label smoothing, mixup, etc

아래 그림은 각 모델을 fine-tuning할 때, Hessian measure가 실제 generalization error와 어떤 관계를 가지는지를 보인다. 유사한 경향성을 보이는걸 알 수 있다.
![](https://i.imgur.com/YYJBFOu.png)

generalization bound를 비교한 결과
![](https://i.imgur.com/NHEsIKr.png)

(다른 실험은 라벨노이즈라 생략)

## Conclusion

##### Conclusion

이 연구는 PAC-Bayesian 분석을 사용하여 깊은 네트워크의 fine-tuning에 대한 일반화를 Hessian을 이용하여 접근했다. Fine-tuning할 때, 초기화로부터의 거리 외에도 헤시안(Hessians)이 일반화를 측정하는데 결정적인 역할을 한다는 것을 보여준다.
이론적으로 깊은 네트워크에서 다양한 fine-tuning 방법에 대한 헤시안 거리 기반 일반화 경계를 정의하였으며, 실제로 Hessian based distance measure가 fine-tuning의 generalization error와 관련있다는 것을 실험적으로 보인다.

##### Future work

- 이 연구에서는 신경망의 activation function이 **두 번 미분 가능해야 한다**는 조건이 존재한다. ReLU와 같은 activatino function에서는 미분 불가능한 점이 발생할 수 있다. (하나의 방법은 미분 불능점이 무시할 수 있는 확률로 발생한다고 주장)
- Hessian이 generalization에 큰 영향을 주기 때문에 다른 블랙박스 모델에서의 Hessian을 이용하여 일반화를 이해하기 위해 사용할 수 있다.
  <br>

[^1]:
    PAC-Bayesian Analysis : Bayesian learning에서 사용되었으며 현재는 일반적인 상황에도 적용가능하다. PAC-Bayes theory gives the tightest known generalization bounds for SVMs
    공부 자료 :

- [leacture_PAC-Bayes analysis](https://courses.cs.washington.edu/courses/cse522/11wi/scribes/lecture13.pdf/)
- [An Introduction to PAC-Bayesian Analysis](https://www.i-aida.org/wp-content/uploads/2021/05/An-Introduction-to-PAC-Bayesian.pdf)
- [ICML 2021 튜토리얼](http://people.kyb.tuebingen.mpg.de/seldin/ICML_Tutorial_PAC_Bayes.htm)

[^2]:
    **Hessian** 어떤 함수의 이계도함수 를 행렬로 표현한 것  
    실함수 $f(x_1, x_2, x_3, \cdots, x_n)$이 주어졌을 때, Hessian Matrix은 다음과 같이 주어진다.
    즉, Hessian Matrix의 $H_{ij}$ 요소는 $x_i$와 $x_j$ 변수에 대해 함수를 이차미분한 값이다.
    ![H(f)={\begin{bmatrix}{\frac  {\partial ^{{2}}f}{\partial x_{{1}}^{2}}}&{\frac  {\partial ^{{2}}f}{\partial x_{{1}}\partial x_{{2}}}}&\cdots &{\frac  {\partial ^{{2}}f}{\partial x_{{1}}\partial x_{{n}}}}{\frac  {\partial ^{{2}}f}{\partial x_{{2}}\partial x_{{1}}}}&{\frac  {\partial ^{{2}}f}{\partial x_{{2}}^{2}}}&\cdots &\vdots \vdots &\vdots &\ddots &\vdots {\frac  {\partial ^{{2}}f}{\partial x_{{n}}\partial x_{{1}}}}&\cdots &\cdots &{\frac  {\partial ^{{2}}f}{\partial x_{{n}}^{2}}}\end{bmatrix}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/745193a2031dc940aac86304231a53e4b75161e3)
    이때, 함수 $f$의 이계도함수가 연속이라면 편미분은 순서상관없이 동일하기에 $f$의 이계도함수가 연속이라면 Hessian Matrix는 대칭행렬(symmetric matrix)이다.  
    기하학적 의미
    _Reference : [헤세 행렬의 기하학적 의미](https://angeloyeo.github.io/2020/06/17/Hessian.html)_
    ![](https://i.imgur.com/Cwer89j.png)
    ![](https://i.imgur.com/7lflpRT.png)

    - 특정 고유벡터에 대해 고유값의 절댓값의 크기가 클수록 해당 방향으로 더 가파르게 변화한다.
    - **헤시안 행렬의 고윳값이 모두 양수라면 함수는 아래로 볼록하다. 일차미분이 0이라면 극솟값** ; convex
    - 헤시안 행렬의 고윳값이 모두 음수라면 함수는 위로 볼록하다. 일차미분이 0이라면 극댓값
    - 헤시안 행렬의 교윳값에 양수와 음수가 섞여있는 경우라면 함수는 안장의 형태. 일차미분이 0이라면 안장점
      > [!note]
      > 즉, 헤시안은 주어진 함수의 곡률을 설명하기 위하여 사용된다. 헤시안은 모델(함수)이 가지는 매개변수에 대한 space서 parameter의 변화에 따라 value가 얼마나 빠르게 변화하는지(즉, 얼마나 '곡률이 높은지' 또는 '곡률이 낮은지')를 알려주는 역할을 수행한다.

[^3]: label noise Learning with noisy labels means When we say "noisy labels," we mean that an adversary has intentionally messed up the labels, which would have come from a "clean" distribution otherwise
[^4]: trace : sum of $X$'s diagonal entres
[^5]: matrix inner product : The matrix inner product is the same as our original inner product between two vectors of length mn obtained by stacking the columns of the two matrices, matrix inner product는 두 행렬의 유사성을 나타내기 위해 사용된다. (내적이 벡터의 유사성을 의미하는 것처럼.)
[^6]:
    \_Lipschitz-continuity : 두 점 사이의 거리를 일정 비 이상으로 증가시키지 않는 함수  
    립시츠 조건을 만족하는 립시츠 연속함수는 균등연속함수이므로 해당 범위에서 미분이 가능하며, 미분 계수는 특정 K 값을 넘을 수 없게 된다.(gradient의 상한이 존재) $$ |f'| \le K $$
    ![Lipchitz|400](https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Lipschitz_Visualisierung.gif/653px-Lipschitz_Visualisierung.gif)

[^7]: Taylor's expansion : 주어진 함수를 그 함수의 도함수(derivatives)를 사용하여 특정 점 주변에서 다항식으로 근사하는 방법
[^8]:
    uniform convergence : 함수의 수열에 대한 수렴성의 특별한 형태 -> 즉, 위의 맥락에서는 loss function을 "모든 sample"에 대해서 계산하기 때문에 각각의 모든 ㅣoss function value가 수렴하는지는 따져보아야함을 의미한다.
    A sequence of functions $\{f_n\}, n = 1, 2, 3, ...$ is said to be uniformly convergent to $f$ for a set $E$ of values of $x$ if, for each $ϵ>0$, an integer $N$ can be found such that $$| f_n(x) - f(x) |< \epsilon$$for $n\ge N$ and all $x\in E$.
    A series $\sum f_n(x)$ converges uniformly on $E$ if the sequence $\{S_n\}$ of partial sums defined by $\sum_{k = 1}^n f_k(x) = S_n(x)$ converges uniformly on $E$.

[^9]: Frobenius norm(=Euclidean norm; in vector) : 행렬의 모든 원소의 절대값의 제곱을 합한 후 그 합의 제곱근
[^10]: statistically-consistent loss : 통계적 일관성이 있는 손실 함수. 데이터의 수가 무한히 증가함에 따라 특정 값에 어떤 확률로 수렴하는 성질을 가지는 loss function
[^11]:
    KL-divergence
    ![](https://i.imgur.com/YxhD6IR.png)
