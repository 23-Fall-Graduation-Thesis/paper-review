# Interpretable Convolutional Neural Networks

---

## Abstract

본 논문에서는 전통적인 CNN을 interpretable CNN으로 수정하는 방법을 제안하여 CNN의 깊은 convolution layer에서의 knowledge representation을 분명히 한다. Interpretable CNN에서, 각 convolution layer의 filter는 특정한 object part를 나타낸다. 본 논문의 interpretable CNN은 지도 학습을 위한 object part나 texture에 대한 주석 처리 없이 일반적인 CNN과 동일한 훈련 데이터를 사용한다. Interpretable CNN은 학습 과정 중에 object part가 있는 깊은 convolution layer에 각 filter를 자동으로 할당한다. 다양한 구조를 가진 서로 다른 종류의 CNN에 본 논문에서 제안하는 emthod를 적용할 수 있다. Interpretable CNN의 명시적인 knowledge representation은 사람이 CNN 내부의 논리를 이해하도록 도울 수 있다. 즉, 예측을 위해 CNN이 어떤 패턴을 기억하는지에 대한 것을 알 수 있다. 실험에 따르면 Interpretable CNN의 filter는 전통적인 CNN의 filter보다 의미론적으로 중요하다. 

코드 링크 : [https://github.com/zqs1022/interpretableCNN](https://github.com/zqs1022/interpretableCNN)

## 1. Introduction

 CNN은 object classification과 detection 같은 많은 visual task에서 뛰어난 성능을 달성해왔다. discrimination power 외에 model interpretability도 신경망의 중요한 특성이다. 그러나 interpretability는 항상 CNN의 아킬레스건 같은 존재였으며, 수십 년 간 어려움을 겪어왔다. 본 논문에서는 *어떤 추가적인 human supervision 없이도 각 conv layer가 interpretable knowledge representation을 얻도록 CNN을 수정할 수 있는가*와 같은 새로운 문제에 초점을 맞춘다. 연구진은 end-to-end learning process가 수행되는 동안 Conv layer의 representation이 특정한 introspection을 가질 것이라고 기대한다. CNN interpretability를 개선하는 문제는 기존 off-line visualization [34, 17, 24, 4, 5, 21] 및 pre-trained CNN representation의 diagnosis [2, 10, 18]와는 다르다. 

 Bau et al. [2]는 CNN의 6가지 semantic 종류를 정의했다: *objects, parts, scenes, textures, materials, and colors.* 사실, 처음 두 semantic을 구체적인 shape를 가진 object-part pattern으로 rough하게 생각할 수 있고, 마지막 4개의 sementic을 명확한 윤곽이 없는 texture pattern으로 요약할 수 있다. 게다가, 얕은 conv layer의 filter는 보통 단순한 texture를 설명하는 반면, 깊은 conv layer의 filter는 object part를 나타낼 가능성이 더 높다. 

![Untitled](./Interpretable%20Convolutional%20Neural%20Networks/fig1.png)

 따라서 본 연구에서는 각 filter를 깊은 conv layer에서 훈련시켜 object part를 표현하는 것을 목표로 한다. Figure 1은 전통적인 CNN과 Interpretable CNN의 차이점을 보여준다. 전통적인 CNN에서 깊은 layer의 filter는 pattern의 혼합을 설명할 수 있는데, 즉 고양이의 머리 부분과 다리 부분에 filter가 활성화될 수 있다. 깊은 conv layer에서의 이러한 복잡한 표현은 네트워크 interpretability를 현저하게 떨어뜨릴 수 있다. 이와 대조적으로 Interpretable CNN에서 filter는 특정 part에 의해 활성화된다. 이러한 방법으로 연구진은 어떤 object part가 모호함[ambiguity] 없이 분류를 위해 CNN filter에 기억될 수 있는지 명시적으로 확인할 수 있었다. 본 연구의 목표는 다음과 같이 요약될 수 있다. 

- 다양한 구조를 가진 CNN에 광범위하게 적용될 수 있는 Interpretability를 개선하기 위해 CNN을 약간 수정할 것을 제안한다.
- supervision을 위한 object part나 texture의 annotation을 필요로 하지 않는다. 대신, 각 filter의 representation을 object part를 향해 자동으로 푸시한다.
- interpretable CNN은 최상위 레이어의 loss function을 변경하지 않으며 원래 CNN과 동일한 training sample을 사용한다.
- 탐색적인 연구로서 interpretability에 대하 설계가 discrimination power를 다소 저하시킬 수는 있지만, 그러한 저하를 작은 범위 내로 제한하고자 한다.

**Method**

 본 논문에서는 CNN의 특정 conv layer에서 filter를 object part의 representation을 향해 push 하는 간단하면서도 효과적인 loss를 제안한다. Figure 2와 같이, 각 filter의 output feature map에 대한 loss를 추가한다. 이 loss는 inter-category의 낮은 엔트로피 neural activation의 공간 분포의 낮은 엔트로피를 장려한다. 즉, 1) 각 filter는 단일 object category에 의해 배타적으로 포함되는 별개의 object part를 encoding 해야 하며, 2) filter는 다른 object part에 반복적으로 나타나는 것이 아니라 객체의 single part에 의해 활성화되어야 한다. 연구진은 다양한 region의 반복적인 shape가 높은 수준의 part 대신 낮은 수준의 texture(color나 edge 등)을 묘사할 가능성이 더 높다고 가정한다. 예를 들어 left eye와 right eye는 두 eye의 context가 symmetric하지만 같지 않기 때문에 다른 filter로 표현될 수 있다. 

![Untitled](./Interpretable%20Convolutional%20Neural%20Networks/fig2.png)

**The value of network interpretability**

 깊은 conv layer에서의 명확한 semantic은 사람들이 네트워크의 예측을 신뢰해야 할 때 매우 중요하다. [38]에서 설명한 바와 같이 dataset bias와 representation bias를 고려하면, 테스트 이미지에 대한 높은 accuracy는 여전히 CNN이 올바른 representation을 인코딩함을 보장할 수 없다. 예를 들면 CNN은 신뢰할 수 없는 context(eye의 features)를 사용하여 얼굴 이미지의 “lipstrick” attribute를 식별할 수 있다. 따라서 CNN이 논리를 의미론적으로[semantically] 혹은 시각적으로 설명할 수 없는 한 인간은 네트워크를 완전히 신뢰할 수 없는 경우가 일반적이다. 이미지가 주어지면 network diagnosis를 위한 현행 연구[5, 21, 18]에서는 pixel level에서 예측 output에 가장 큰 기여를 하는 이미지 region을 localize한다. 본 연구에서는 CNN이 object-part 부분 level에서 logic을 설명할 거라고 예상한다. Interpretable CNN이 주어지면 object classification를 위해 CNN이 기억하는 object part의 distribution을 명시적으로 보여줄 수 있다. 

**Contributions**

 본 논문에서는 새로운 task, 즉 깊은 conv-layer에서의 representation이 interpretable한 CNN을 end-to-end learning 하는 것에 초점을 맞추고 있다. 연구진은 supervision을 위한 object part나 texture의 어떠한 주석 처리도 없이 다양한 유형의 CNN을 interpretable CNN으로 수정할 수 있는 간단하지만 효과적인 mehod를 제안한다. 실험에 따르면 이러한 접근 방식이 CNN의 object-part interpretability를 크게 향상시켰다. 

## 2. Related work

 연구진의 이전 논문[39]은 

1) CNN representation의 visualization 및 diagnosis

2) CNN representation을 그래프나 트리로 분해하기 위한 접근 방식

3) 분해되고 interpretable한 representation을 가진 CNN의 학습

4) model interpretability를 기반으로 한 middel-to-end learning

을 포함하여 신경망의 visual interpretability를 탐구하는 최근 연구에 대한 포괄적인 조사를 제공한다. 

**Network visualization**

 CNN에서 filter의 visualization은 neural unit 내부에 숨겨진 패턴을 탐색하는 가장 직접적인 방법이다. [34, 17, 24]는 주어진 unit의 score를 최대로 하는 외관[appearance]를 보여주었다. up-convolutional nets [4]는 CNN feature map을 image로 변환하기 위해 사용되었다. 

**Pattern retrieval**

 일부 연구는 수동적 시각화[passive visualization]을 넘어 다양한 application에 대해 CNN으로부터 특정 unit을 능동적으로 탐색[actively retrieve]한다. 이미지에서 mid-level feature의 추출[26]과 마찬가지로 pattern retrieval은 주로 conv layer로부터 mid-level representation을 학습한다. Zhou et al. [40, 41]은 “scenes”을 설명하기 위해 feature map으로부터 unit을 선택했다. Simon et al. [22]은 label이 지정되지 않은 이미지의 feature map에서 object를 발견하고 특정 filter를 선택하여 각 semantic part를 supervision 방식으로 설명했다. Zhang et al. [36]은 약하게 지도하는 방식으로 object part를 표현하기 위해 filter의 feature map으로부터 특정 neural unit을 추출했다. 그들은 또한 active question-answering [37]을 통해 Andd-Or graph로 CNN representation을 분리했다. [11, 31, 29, 15]는 다양한 application을 위해 CNN의 특정한 의미를 가지는 neural unit을 선택했다. 

**Model diagnosis**

 CNN의 feature를 분석하기 위해 많은 statistical method [28, 33, 1]들이 제안되었다. Ribeiro et al [18]이 제안한 LIME method, influence functions [10] 및 gradient-based visualization methods [5, 21]과 [13]은 network rerpesentation을 해석하기 위해 각 network output을 담당하는 이미지 region을 추출했다. 이러한 방법은 살마들이 각 테스트 이미지의 label prediction을 담당하는 이미지 region을 수동으로 확인해야 한다. [9]는 CNN에서 다양한 category의 representation 간 관계를 추출했다. 이와는 대조적으로 interpretable CNN이 주어지면 사람들은 inference 절차 중 결정에 사용되는 object parts (filters)을 직접 식별할 수 있다. 

**Learning a better representation** 

pre-trained CNN의 diagonosis and/or visualization과 달리, 일부 접근 방식은 더 의미있는 representation을 학습하도록 개발되었다. [19]는 더 나은 모델을 학습하기 위해 사람들이 각 output과 관련된 input의 label을 dimension으로 지정할 것을 요구했다. Hu et al [8]은 네트워크 output에 대한 몇 가지 logic rule을 설계하고 이러한 rule들을 사용하여 학습 프로세스를 정규화했다. Stone et al. [27]은 더 나은 object compositionality로 CNN representation을 학습했고, Liao et al. [16]에서는 compact한 CNN representation을 학습했지만 filter가 명시적인 part-level이나 texture-level semantic을 얻도록 만들지는 않았다. Sabour et al. [20]은 dynamic routing mechanism을 사용하여 전체 object를 capsule의 parsing tree로 파싱하고, 각 capsule은 특정한 의미를 인코딩할 수 있는 capsule model로 제안했다. 본 논문에서는 filter의 interpretability를 향상시키기 위해 filter의 representation을 정규화하기 위한 generic loss를 만들어냈다. 

 연구진은 information bottleneck [32]의 관점에서 Interpretable CNN을 다음과 같이 이해할 수 있다. 

1) 연구진의 interpretable filter는 conv layer의 feature map이 주어졌을 때 최종 classification의 조건부 엔트로피가 최소화되도록 각 category의 가장 구별되는[distinct] part를 선택적으로 모델링한다. 

2) 각 filter는 object의 single part를 나타내며, 이는 input image와 middle-layer feature map 간 mutual inforlation을 최대화한다. (즉, 관련 없는 information은 가능한 많이 “forgetting”한다.)

## 3. Algorithm

 CNN의 target conv layer가 주어지면, 우리는 conv layer의 각 filter가 특정 category의 특정한 object part에 의해 활성화되는 반면, 다른 category들의 이미지에 대해서는 비활성된 상태로 유지될 것이라고 예상할 수 있다. 훈련 이미지 데이터셋을 $I$라고 하자. 여기서 $I_c \subset I$는 $(c=1,2,…C)$ 중 category $c$에 속하는 부분 집합을 나타낸다. 이론적으로, multi-class classification, single-class classification (즉, category의 이미지에 대해서는 $c=1$, random 이미지에 대해서는 $c=2$) 및 기타 task에 대해 CNN을 학습시키기 위해 다양한 유형의 loss를 사용할 수 있다. 

 Figure 2는 interpretable conv layer의 구조를 보여준다. 다음 단락에서는 target conv layer에서 single filter $f$의 학습에 초점을 맞춘다. 연구진은 ReLU 연산 후에 filter $f$의 feature map $x$에 loss를 추가한다. 

**During the forward propagation**

 각 입력 이미지 $I$가 주어지면, CNN은 ReLU 연산 후 filter $f$의 feature map $x$를 계산하며, 여기서 $x$는 $n \times n$ 행렬이고 각 요소 $x_{i,j} \geq 0$이다. 본 논문의 method는 feature map $x$에서 object part의 potential position을 가장 강한 activation $\hat{\mu} = \text{arg max}_{\mu=[i,j]} x_{ij}, 1 \leq i,j \leq n$를 갖는 neural unit으로 추정한다. 그런 다음 추정된 part position $\hat{\mu}$를 기반으로 CNN은 noisy activation을 필터링하기 위해 특정 mask를 $x$로 할당한다. 

 $f$에 대응하는 object part는 서로 다른 이미지가 주어지면 $n^2$개의 서로 다른 위치에 나타날 수 있기 때문에, 연구진은 $f, \{ T_{\mu_1}, T_{\mu_2}, \cdots ,T_{\mu_{n^2}} \}$에 대한 $n^2$개의 템플릿을 설계한다. Figure 3에 보이는 바와 같이, 각 템플릿 $T_{\mu_i}$는 $n \times n$ 행렬이며, target part가 $x$의 $i$번째 unit을 주로 트리거할 때 feature map $x$에 대한 이상적인 activation distribution을 표현한다. 연구진의 method는 $n^2$개의 템플릿으로부터 part position $\hat{\mu}$과 관련된 템플릿 $T_{\hat{\mu}}$을 선택한다. 연구진은 $x^{\text{masked}} = \text{max} \{ x \circ T_{\hat{\mu}}, 0 \}$를 output masked feature map으로 계산하고, 여기서 $\circ$는 Hadamard (element-wise) product를 의미한다. 

![Untitled](./Interpretable%20Convolutional%20Neural%20Networks/fig3.png)

 Figure 4는 서로 다른 이미지에 대해 선택된 mask $T_{\hat{\mu}}$를 visualize하고 원본과 마스킹된 feature map을 비교한다. CNN은 서로 다른 이미지에 대해 서로 다른 템플릿을 선택한다. mask 연산은 gradient back-propagation을 지원한다. 

![Untitled](./Interpretable%20Convolutional%20Neural%20Networks/fig4.png)

**During the back-propagation process**

 본 논문의 loss는 filter $f$가 특정한 category $c$의 object part를 나타내고 다른 category의 이미지에 대해서는 별 반응을 보이지 않도록 하는 방향으로 푸시한다. filter $f$에 대한 category $c$의 결정에 대해서는 3.1절을 참조하라. $\bold X = \{ x | x = f(I), I \in \bold I \}$가 서로 다른 훈련 이미지에 대해 ReLU 연산을 한 후 $f$의 feature map을 나타낸다고 하자. 입력 이미지 $I$가 주어졌을 때, $I \in \bold  I_c$라면 feature map $x = f(I)$가 target part의 위치에 배타적으로 활성화될 것임을 예상할 수 있다. 반대의 경우 feature map은 비활성화된 상태를 유지할 것이다. 즉, $I \in \bold I_c$인 경우 feature map $x$는 할당된 템플릿 $T_{\hat{\mu}}$와 딱 맞을 거라고 예상할 수 있다. 반대로 $I \notin \bold I_c$인 경우 negative 템플릿 $T^{-}$을 설계하고 feature map $x$가 $T^{-1}$과 맞길 바랄 것이다. forward propagation이 수행되는 동안, 연구진의 method는 negative 템플릿을 생략함을 유의해야 한다. 다른 category들에 대한 것들을 모두 포함하는 모든 feature map은 $n^2$개의 템플릿 $\{ T_{\mu_i} \}$을 mask로 선택한다. 

 따라서, 각 feature map $x$는 $n^2+1$개의 템플릿 $T \in \bold T = \{ T^{-1}, T_{\mu_1}, T_{\mu_2}, \cdots ,T_{\mu_{n^2}} \}$ 중에서 하나에 배타적으로 적합하도록 가정된다. 연구진은 $f$에 대한 loss를 $\bold X$와 $\bold T$ 간 minus mutual information, 즉 $-MI(\bold X ; \bold T)$로 공식화한다. 

$$
\textbf{Loss}_f = -MI(\bold X; \bold T) = -\sum_Tp(T) \sum_xp(x|T) \ \text{log } \frac{p(x|T)}{p(x)} \tag{1}
$$

템플릿의 사전 확률[prior probability]는 $p(T_\mu) = \frac{\alpha}{n^2}, p(T^{-1}) = 1 - \alpha$로 주어지며, 여기서 $\alpha$는 일정한 사전 우도[prior likelihood]이다. feature map $x$와 템플릿 $T$ 사이 적합도는 조건부 우도 [conditional likelihood] $p(x|T)$로 측정된다. 

$$
\forall T \in \bold T, \ \ \ \ \ p(x|T) = \frac{1}{Z_T} \text{exp} \left[ tr(x \cdot T) \right] \tag{2}
$$

여기서 $Z_T = \sum_{x \in \bold X} \text{exp }\left[ tr(x \cdot T) \right]$이다. $x \cdot T$는 $x$와 $T$ 사이 multiplication을 나타내고, $tr(\cdot)$은 matrix의 trace를 나타내며, $tr(x \cdot T) = \sum_{ij} x_{ij} t_{ij}$이다. $p(x) = \sum_T p(T) p(x|T)$이다. 

**Part templates**

Figure 3에 나오는 바와 같이, negative template은 $T^{-1} = (t_{ij}^{-}), t_{ij}^{-} = -\tau < 0$으로 주어지며 $\tau$는 양의 상수이다. $\mu$에 대응하는 positive template은 $T_\mu = (t_{ij}^+), t_{ij}^+ = \tau \cdot \text{max } (1 - \beta \frac{\left\| [i,j] - \mu \right\|_1}{n}, -1)$으로 주어지며, 여기서 $\left\| \cdot \right\|_1$은 $L_1$ norm distance이고 $\beta$는 constant parameter이다. 

![Untitled](./Interpretable%20Convolutional%20Neural%20Networks/fig3.png)

### 3.1 Learning

 end-to-end 방식을 통해 interpretable CNN을 훈련시킨다. forward propagation process 중에, CNN의 각 filter는 전통적인 CNN과 같은 방식으로 상향식 방식으로 information을 전달한다. backward propagation process 중에, interpretable conv layer의 각 filter는 k번째 샘플의 최종 task loss $\bold L(\hat{y}_k, y_k^*)$와 filter loss $\textbf{Loss}_f$ 모두로부터 feature map $x$에 대한 gradient를 다음과 같이 받는다. ($\lambda$는 weight)

$$
\frac{\partial \textbf{Loss}}{\partial x_{ij}} = \lambda \frac{\partial \textbf{Loss}}{\partial x_{ij}} + \frac{1}{N} \sum^N_{k=1} \frac{\partial \ \bold L( \hat{y}_k, y_k^*)}{\partial x_{ij}} \tag{3}
$$

 그러고 나서 얕은 레이어로 $\frac{\partial \textbf{Loss}}{\partial x_{ij}}$를 역전파하고 CNN을 업데이트하기 위해 feature map과 얕은 레이어의 parameter와 관련된 Xgradient에 대한 gradient를 계산한다. 

 구현을 위해, feature map $x$의 각 요소 $x_{ij}$와 관련된 $\textbf{Loss}_f$의 gradient를 다음과 같이 계산한다. 

$$
\frac{\partial \textbf{Loss}}{\partial x_{ij}} = \frac{1}{Z_T} \sum_T p(T) t_{ij} e^{tr(x \cdot T)} \{ tr(x \cdot T) - log \left[ Z_T p(x) \right] \} \\ \approx \frac{p(\hat{T})\hat{t}_{ij}}{Z_{\hat{T}}} e^{tr(x \cdot \hat{T})} \{ tr(x \cdot \hat{T}) - log Z_{\hat{T}} - log\ p(x) \} \tag{4}
$$

여기서 $\hat{T}$는 feature map $x$에 대한 target template이다. 만약 주어진 이미지 $I$가 filter $f$의 target category에 속한다면, $\hat{T} = T_{\hat{\mu}}$이고, 여기서 $\hat{\mu} = \text{arg max}_{\mu=[i,j]} x_{ij}$이다. 만약 이미지 $I$가 다른 catefory에 속한다면, $\hat{T} = T^{-}$이다. 최초의 학습 에피소드 후에 $\forall T \in \bold T \setminus \{ \hat{T} \}$, $e^{tr(x \cdot \hat{T})} \gg e^{tr(x \cdot T)}$를 고려하여 계산을 단순화하기 위해 위의 근사치를 만든다. $Z_T$는 수많은 feature map을 사용하여 계산되기 때문에 위의 식에서 gradient를 계산하기 위해 $Z_T$를 대략적인 상수로 취급할 수 있다. 연구진은 훈련 과정 동안 $Z_T$의 값을 점진적으로 옵데이트한다. 마찬가지로 $p(x)$도 방대한 계산 없이 근사화할 수 있다. 

**Determining the target category for each filter**

target category $\hat{c}$를 갖는 각 filter $f$를 Eq. 4의 근사 gradient에 할당해야 한다. 본 논문에서는 단순하게 이미지가 $f$를 가장 많이 활성화하는 catefory $\hat{c}$를 갖는 filter $f$를 할당한다. 

즉, $\hat{c} = \text{arg max}_c \text{mean}_{x = f(I):I \in I_C} \sum_{ij} x_{ij}$

## 4. Understanding of the loss

사실 Eq.1의 loss는 다음과 같이 다시 쓸 수 있다. 

![Untitled](./Interpretable%20Convolutional%20Neural%20Networks/lossf.png)

여기서 $\bold T' = \{ T^{-1}, \bold T^+ \}$이다. $H(\bold T) = -\sum_{T \in \bold T} p(T) log \ p(T)$는 part template의 constant prior entropy이다. 

**Low inter-category entropy**

두 번째 term $H(\bold T' = \{T^{-}, \bold T^+ \} |\bold X)$는 다음과 같이 계산된다. 여기서 $\bold T^+$는 $\bold T^+ = \{ T_{\mu_1}, \cdots ,T_{\mu_{n^2}} \subset \bold T$이고 $p(\bold T^+|x) = \sum_\mu p(T_\mu | x)$이다. 

![Untitled](./Interpretable%20Convolutional%20Neural%20Networks/H.png)

모든 positive template $\bold T^+$의 집합은 category $c$를 나타내기 위한 single label로 정의한다. negative template $T^{-}$는 다른 category를 나타낼 때 사용한다. 이 term은 category 내 activation의 조건부 엔트로피가 낮아지도록 격려한다. 즉, 잘 훈련된 filter $f$는 특정 category $c$에 의해 배타적으로 활성화되고 다른 category에 대해서는 반응이 없도록 해야 한다. $f$의 feature map $x$는 입력 이미지가 category $c$에 속하는지 아닌지 확인하는 데에 사용될 수 있다. 즉, $x$는 엄청난 불확실성 없이 $T_{\hat{\mu}}$나 $T^{-}$ 중 하나로 fitting 된다. 

**Low spatial entropy**

 Eq. 5의 세 번째 term은 다음과 같이 주어지며, $\tilde{p}(T_\mu | x) = \frac{p(T_\mu | x)}{p(\bold T^+ | x)}$이다. 

![Untitled](./Interpretable%20Convolutional%20Neural%20Networks/Hp.png)

이 term은 $x$의 활성화되는 공간 분포의 조건부 엔트로피가 낮아지도록 격려한다. 즉, 이미지 $I \in \bold I_c$가 주어지면, 잘 학습된 filter는 다른 위치에서 각각 트리거되는 대신 feature map $x$의 single region $\hat{\mu}$에 의해서만 활성화되어야 한다.  

## 5. Experiments

 실험에서 해당 메소드가 널리 적용됨을 증명하기 위해 연구진은 그들의 method를 4가지의 CNN 종류에 적용했다. 또한, 단일 카테고리 분류 및 다중 카티고리 분류에 대한 interpretable CNN을 학습하기 위해 3가지의 서로 다른 벤치마크 데이터셋에서 object image를 사용했다. 그리고 filter들의 semantic 적인 의미를 설명하기 위해 conv layer에서 filter들의 feature map들을 시각화했다. 연구진은 2가지 타입의 metric(object-part interpretability 및 location instability)를 사용하여 filter의 part semantic이 명확함을 검증했다. 실험 결과는 연구진의 filter들이 기존 CNN에서보다 interpretable CNN에서 훨씬 더 semantic 적으로 의미있음을 보여준다. 

**Three benchmark datasets**

 각 filter의 semantic 적인 명확성을 평가하기 위해 object landmark (part)에 대한 ground-truth annotation이 필요했기 때문에, ILSVRC 2014 Det Animal-Part dataset [36], CUB200-2011 dataset [30], Pascal VOC part dataset [3] 등을 포함하여 학습과 테스트를 위해 part annotation이 있는 세 개의 벤치마크 데이터셋을 선택했다. [3, 36]에서 설명한 바와 같이, 동물 카테고리에 경직되지 않은 부분은 일반적으로 part localization에 엄청난 문제를 제시한다. 따라서 평가를 위해 세 개의 데이터셋에서 37개의 동물 마테고리를 선택하기 위해 [3, 36]을 따라 3개의 데이터셋에서 37개의 동물 범주를 선택했다. 

 세 데이터셋 모두 전체 object의 ground-truth bounding box들을 제공한다. landmark annotation의 경우, ILSVRC 2013 DET Animal-Part dataset [36]에는 30개의 동물 카테고리의 머리와 다리의 groun-truth bounding box들이 포함되어 있다. CUB200-2011 dataset [30]에는 200 종의 총 11.8K개의 새 이미지가 포함되어 있으며 데이터셋은 15개의 새 landmask annotation을 제공한다. Pascal VOC Part dataset [3]에는 6갸의 동물 카테고리의 107개 object landmark의 ground-truth part segmentation이 포함되어 있다. 

**Four types of CNNs**

 연구진은 4개의 일반적인 CNN, 즉 AlexNet [12], VGG-M [25], VGG-s [25], VGG-16 [25] 등을 interpretable CNN으로 수정했다. residual network [7]의 skip connection은 일반적으로 다른 filter의 single feature map 인코딩 패턴을 만든다는 점을 고려하여, 본 연구에서는 스토리를 단순화하기 위해 residual network에서 최상위 conv-layer 성능을 테스트하지 않았다. 특정 CNN이 주어지면 원래 네트워크의 최상위 conv layer에 있는 모든 filter를 해석 가능한 filter로 수정한다. 그런 다음 원래 최상위 conv layer 위에 $M$개의 filter가 있는 conv layer를 삽입했고, 여기서 $M$은 새로운 conv layer 입력의 채널 개수이다. 연구진은 새로운 conv layer의 filter를 interpretable filter로 설정했다. 각 filter는 bias 항이 있는 $3 \times 3 \times 4$ 텐서였다. 연구진은 출력 feature map과 input이 동일한 크기인지 확인하기 위해 feature map에 zero padding을 추가했다. 

**Implementation details**

 파라미터는 다음과 같이 설정했다: $\tau = \frac{0.5}{n^2}, \alpha = \frac{n^2}{1+n^2}, \beta = 4$. 온라인 방식으로 neural activation의 크기에 대한 filter loss의 weight를 업데이트했으며, 여기서 $t$번째 에포크에 $\lambda \propto \frac{1}{t} \text{mean}_{x \in \bold X} \text{max}_{i,j}x_{ij}$이다. fc layer와 새로운 conv layer의 파라미터를 초기화하고 [12, 25]에서 1.2M ImageNet 이미지를 사용하여 사전학습된 전통적인 CNN에서 다른 conv layer들의 파라미터를 로드했다. 그런 다음 데이터셋의 훈련 이미지를 사용하여 interpretable CNN의 모든 layer의 파라미터를 fine-tuning한다. 공정한 비교를 가능하게 하기 위해 기존 CNN은 fc layer의 파라미터를 초기화하고 conv layer 파라미터를 로드함으로써 fine-tuning도 했다. 

### 5.1 Experiments

**Single-category classification**

 ILSVRC 2013 DET Animal-Part dataset [36], CUB200-2011 dataset [30] 및 Pascal VOC Part dataset [3]에서 각 카테고리를 분류하기 위해 AlexNet, VGG-M, VGG-S 및 VGG-16 구조를 기반으로 네 가지 유형의 interpretable CNN을 학습했다. 또한, 비교를 위해 동일한 데이터를 사용하여 일반적인 AlexNet, VGG-M, VGG-S 및 VGG-16 네트워크도 학습시켰다. single-category classification을 위해 logistic log loss를 사용했다. [36, 35]의 실험 설정에 따라, bounding box에 기초하여 target category의 object를 ground-truth labels $y^* = +1$을 가진 positive sample로 잘라냈다. 다른 범주의 이미지면 ground-truth label $y^* = -1$을 갖도록 negative sample로 간주했다. 

**Multi-category classification**

 multi-category classification을 위한 CNN 학습을 위해 Pascal VOC Part dataset [3]에서 6개의 동물 카테고리와 ILSVRC 2013 DET Animal-Part dataset [36]에서 30개의 카테고리를 각각 사용했다. 그리고 VGG-M, VGG-S 및 VGG-16 구조를 기반으로 interpretable CNN을 학습했다. multi-class classification을 위한 softmax log loss와 logistic loss loss의 두 가지 타입의 loss를 사용해보았다. 

### 5.2 Quantitative evaluation of part interpretablity

 [2]에서 논의한 바와 같이, 얕은 conv layer의 filter는 일반적으로 단순한 패턴이나 object detail(ex. edges, simple textures, and colors)을 나타내는 반면, 깊은 conv layer의 filter는 복잡하고 큰 규모의 part를 나타낼 가능성이 더 높다. 따라서 실험에서 CNN의 최상위 conv layer에 대한 part semantic의 명확성을 평가했다. 평가를 위해 다음 2개의 metric을 사용했다. 

**5.2.1 Evaluation metirc: part interpretability**

 filter의 object part interpretability를 측정하기 위해 Bau et al. [2]에서 제안한 metric을 따른다. 이 evaluation metric을 다음과 같이 간략하게 소개할 것이다. 각 filter $f$에 대해, 서로 다른 input image에 대한 ReLU/mask operation 후에 feature map $\bold X$에 대해 계산했다. 그러고 나서 계산된 모든 feature map의 모든 position에서의 activation의 distribution을 계산한다. [2]는 모든 faeture map $x \in \bold X$에 대한 모든 공간적인 위치 $[i, j]$로부터 최상위 activation을 $f$의 semantic에 상응하는 valid map region으로 선택할 수 있도록 activation threshold $T_f$를 $p(x_{ij} > T_f) = 0.005$가 되도록 설정했다. 그런 다음 [2]는 low-resolution valid map region을 image resolution으로 확장하여 각 image에서 valid activation의 receptive field (RF)를 획득했다. $S^I_f$로 표기되는 이미지 $I$의 RF는 $f$의 part region을 나타낸다. 

 각 filter $f$와 이미지 $I$의 $k$번째 part 간 호환성[compatibility]은 intersection-over-union score $IoU^I_{f,k} = \frac{\left\| S^I_f \cap S^I_k \right\|}{\left\| S^I_f \cup S^I_k \right\|}$으로 매겨질 수 있으먀, 여기서 $S^I_k$는 이미지 $I$에 대한 $k$번째 part의 ground-truth mask를 나타낸다. 연구진은 이미지 $I$가 주어졌을 때 $IoU^I_{f,k} > 0.2$인 filter $f$의 $k$번째 part와 연관시켰다. 부분 연관성[part association]에 대한 $IoU^I_{f,k} > 0.2$의 기준은 [2]에서 사용된 $IoU^I_{f,k} > 0.04$보다 더 엄격하다. 이는 [2]에서 설명한 다른 CNN semantic(color와 texture 같은)과 비교할 때 object-part semantic이 더 엄격한 기준을 요구하기 때문이다. 연구진은 $k$번째 part가 filter $f$E와 연관될 확률은 $P_{f,k} = \text{mean}_{I \text{:with k-th part}} \bold 1 (IoU^I_{f,k} > 0.2)$으로 계산했다. 하나의 filter는 이미지에서 여러 object part와 연관될 수 있음을 유의해야 한다.  연구진은 모든 part 중에서 filter $f$, 즉 $P_f = \text{max}_k P_{f,k}$의 interpretability로 가장 높은 확률의 부분 연관성을 기록했다. 

**For single-category classification**

 single-category classification에서 evaluation을 하기 위해 target category의 테스트 이미지들을 사용했다. Pascal VOC Part dataset [3]에서, bird category에 4개의 part를 사용했다. 연구진은 머리, 부리, 왼쪽/오른쪽 눈을 head part로 병합하고, 몸통, 목, 왼쪽/오른쪽 날개 부분을 torso[몸통] part로 병합하고, 왼쪽/오른쪽 다리와 발 영역을 leg part로 병합하고, 꼬리 부분을 네 번째 part로 사용했다. cat category에 대해서는 5개의 part를 사용했다. 머리, 왼쪽/오른쪽 눈, 왼쪽/오른쪽 귀, 코의 영역을 head parf로 병합, 몸통과 목의 영역의 torso part로 병합, 앞 왼/오 다리와 발 영역을 frontal legs part로 병합, 뒷 왼/오 다리와 발을 back legs part로 병합하고, 꼬리를 마지막 5번째 part로 사용했다. 연구진은 고양이 범주와 유사한 방식으로 cow에 대해서도 네 개의 part를 사용했다. 여기서는 head part에 왼/오 뿔을 추가하고 꼬리 부분을 생략했다. dog의 범주는 고양이 범주와 동일하게 적용했다. 말과 양은 소와 동일한 방식으로 네 개의 part를 적용했다. 평가를 위해 모든 filter에 대해 average part interpretability $P_f$를 계산했다. 

**For multi-category classification**

 먼저 각 filter $f$에 target category $\hat{c}$를 할당한다. 즉, 그 category는 filter를 가장 많이 활성화한 category이다: $\hat{c} = \text{argmax}_c \textbf{mean}_{x:I \in \bold I_c} \sum_{i,j} x_{ij}$. 그러고 나서 앞서 소개된 것처럼, category $\hat{c}$의 이미지들을 사용하여 object-part의 interpretability를 계산한다. 

**5.2.2 Evaluation metric: location stability**

 두 번째 metric은 [35]에서 제안된 part location의 stablity를 측정한다. filter $f$의 feature map $x$가 주어졌을 때, 가장 많이 활성화된 unit $\hat{\mu}$를 $f$의 location inference로 간주했다. $f$가 다른 object를 통해 동일한 object part를 일관되게 표현했다면, 추론된 part location $\hat{\mu}$와 일부 object landmark 사이의 distance는 서로 다른 object part 간에 대해서는 많이 변하지 않아야 한다고 가정했다. 예를 들어 $f$가 어깨를 표현했다면, 어깨와 머리 사이 distance는 서로 다른 object에 대해 안정적으로 유지되어야 한다. 

 따라서 [35]는 추론된 position $\hat{\mu}$와 서로 다른 이미지 중 특정 ground-truth landmark 간 distance의 deviation을 계산하고, $f$의 location stability를 평가하기 위해 다양한 landmark에 대한 average deviation을 사용했다. deviation이 작을수록 location statbility가 높음을 나타낸다. $d_I(p_k, \hat{\mu}) = \frac{\left\| \bold p_k - \bold p(\hat{\mu}) \right\|}{\sqrt{w^2 + h^2}}$가 이미지 $I$에 대해 추론된 part와 $k$번째 landmark $\bold p_k$ 사이 정규화된 distance를 나타내도록 하고, 여기서 $\bold p(\hat{\mu})$는 이미지 평면 상 RF를 역전파 했을 때의 unit $\hat{\mu}$의 RF의 중심을 나타낸다. $\sqrt{w^2 + h^2}$는 입력 이미지의 diagonal length를 나타낸다. 연구진은 k번째 landmark에 대한 filter $f$의 *relative location deviation* 으로 $D_{f,k} = \sqrt{\text{var}_I[d_I(p_k, \hat{\mu})]}$를 계산한다. 여기서 $\text{var}_I[d_I(p_k, \hat{\mu})]$는 distance $d_I(p_k, \hat{\mu})$의 variation이다. 각 landmark가 모든 테스트 이미지에서 나타나 수는 없기 때문에 각 filter $f$에 대해 $k$번째 landmark가 포함된 이미지에 상위 100개의 activation score $x_{\hat{\mu}}$를 갖는 추론 결과만을 사용하여 $D_{f,k}$를 계산했다. 따라서 모든 landmark에서 모든 filter의 relative location deviation의 평균, 즉 $\text{mean}_f \text{mean}^K_{k=1} D_{f,k}$를 사용하여 $f$의 location instability를 측정했다. 여기서 $K$는 landmark의 개수를 나타낸다. 

 구체적으로 각 category의 object landmark를 다음과 같이 선택했다. ILSVRC 2013 DET Animal-Part Dataset [36]의 경우 각 category의 *haed*와 *frontal legs*를 evaluation용 landmark로 사용했다. Pascal VOC Part Dataset [3]의 경우 각 category의 *head*, *neck* 및 *torso[몸통]*를 landmark로 선택했다. CUB200-2011 Dataset [30]의 경우 새의 head, back, tail의 ground-truth position을 landmark로 사용했다. 이 landmark들이 테스트 이미지에 가장 많이 나타났기 때문이다. 

 multi-category classification을 위해 각 filter $f$에 대해 두 개의 항, 즉 1) $f$가 주로 표현하는 category와 2) $f$의 target category의 landmark에 대한 relative position deviation $D_{f,k}$를 결정해야 한다. 기존 $CNN$의 filter들은 single category를 배타적으로 표현할 수 없기 때문에, 연구진은 filter $f$에 landmark가 가장 낮은 location deviation을 달성하도록 하는 category를 할당하여 계산을 단순화했다. 즉, average location deviation $\text{mean}_f \ \text{min}_c \ \text{mean}_{k \in Part_c} D_{f,k}$를 사용하여 location stability를 평가했으며, $Part_c$는 category $c$에 속하는 part index의 집합을 나타낸다. 

**5.2.3. Experimental results and analysis**

 Table 1과 Table 2는 single-category classification일 때 CNN의 part interpretability를, multi-category classification일 때 CNN의 part interpretablity를 각각 비교한다. 

![Untitled](./Interpretable%20Convolutional%20Neural%20Networks/table1.png)

![Untitled](./Interpretable%20Convolutional%20Neural%20Networks/table2.png)

Table 3, 4, 5는 single-category classification일 때 CNN의 average relative location deviation을 나열한다. 

![Untitled](./Interpretable%20Convolutional%20Neural%20Networks/table3.png)

![Untitled](./Interpretable%20Convolutional%20Neural%20Networks/table4.png)

![Untitled](./Interpretable%20Convolutional%20Neural%20Networks/table5.png)

Table 6은 multi-category classification일 때 CNN의 average relative location deviation을 비교한다. 본 논문의 interpretable CNN은 거의 모든 비교에서 기존 CNN보다 훨씬 높은 interpretability와 훨씬 더 나은 location stability를 보여주었다. Table 7은 서로 다른 CNN의 classification accuracy를 비교한다. 기존 CNN은 single-category classification에서 더 우수한 성능을 보이는 반면, multi-category classification의 경우 interpretable CNN이 기존 CNN보다 우수한 성능을 보여주었다. multi-category classification의 좋은 성능은 초기 에포크에 filter semantic의 clarification을 명확히 함으로써 filter learning의 어려움을 줄였기 때문일 수 있다. 

![Untitled](./Interpretable%20Convolutional%20Neural%20Networks/table6.png)

![Untitled](./Interpretable%20Convolutional%20Neural%20Networks/table7.png)

### 5.3. Visualization of filters

 Zhou et al. [38]에서 제안한 방법에 따라 interpretable filter와 neural activation의 RF를 계산했으며, 이를 image resolutoin까지 확장했다. Figure 5는 single-category classification을 위해 훈련된 CNN의 상위 conv-layer에서 filter들의 RF를 보여준다. interpretable CNN의 filter는 주로 특정 object part에 의해 활성화된 반면, 일반적인 CNN의 filter는 대개 명시적인 의미를 갖지 않았다. 

![Untitled](./Interpretable%20Convolutional%20Neural%20Networks/fig5.png)

 Figure 6은 interpretable filter로 인코딩된 object part의 distribution에 대한 heatmap을 보여준다. interpretable filter는 보통 category의 뚜렷하게 구분되는 object part를 선택적으로 모델링하고 다른 part는 무시한다. 

![Untitled](./Interpretable%20Convolutional%20Neural%20Networks/fig6.png)

## 6. Conclusion and discussions

 본 논문에서는 전통적인 CNN의 interpretability를 높이기 위해 이를 수정하는 general method를 제안했다. [2]에서 논의한 바와 같이, discrimination power 외에도 interpretability는 네트워크의 또 다른 중요한 특성이다. 연구진은 supervision을 위한 추가 annotation 없이 깊은 conv-layer에서 filter를 object part의 representation에 가까워지도록 하는 loss를 설계했다. 실험 결과, interpretable CNN이 전통적인 CNN보다 깊은 conv-layer에서 의미 있는 지식을 더 많이 인코딩하는 것으로 나타났다. 

 향후 연구에서는 더 높은 모델 유연성을 달성하기 위해 category의 discriminative texture를 표현하는 새로운 filter와 여러 category가 공유하는 object part에 대한 새로운 filter를 설계할 예정이다.