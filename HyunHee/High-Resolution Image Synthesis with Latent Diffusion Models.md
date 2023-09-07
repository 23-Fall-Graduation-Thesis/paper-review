# High-Resolution Image Synthesis with Latent Diffusion Models

---

# Abstract

### 기존 DM (Diffusion Model)

---

image formation process를 denoising autoencoder의 연속적인 적용으로 분해

**장점**

- 이미지 데이터 뛰어넘는 SOTA 합성 결과 달성
- formulation을 통해 재훈련 없이 이미지 생성 프로세스를 다루는 메커니즘 가이드 가능

**단점**

- 대부분 pixel space 기반 
→ DM optimization에 대한 수백 GPU days와 연속적인 evaluation으로 인한 비싼 추론 비용 발생

### New LDM (Latent diffusion model)

---

pretrained autoencoder의 latent space에서 연산을 적용

**특징**

- 복잡도를 감소시키고 세부사항을 유지하면서 한 번의 연산으로 optimal에 가까운 point에 도달하도록 함
- 모델 아키텍처로 cross-attention layer 도입

**장점**

- diffusion model이 text나 bounding box 같은 general conditioning input에 강력/유연해지게 하며,
    
    high resolution 합성이 convolution 방식으로 가능하게 함
    
- pixel-based DM과 비교해 계산 비용 상당량 감소
- image inpainting과 class-conditional image synthesis에 대한 새로운 SOTA score 달성
- text-to-image synthesis, unconditional image generation 및 super-resolution을 포함한 다양한 task에 대해 매우 경쟁력 있는 성능

### Introduction Summary

---

본 논문에서는 Diffusion Model 학습을 아래의 두 phase로 나누어 분석

- perceptual compression : high-frequency detail 제거; semantic variation은 거의 학습하지 않음
- semantic compression : 실제 생성 모델이 semantic/conceptual한 data의 구성을 학습
- 

![Untitled](./image/Untitled%202.png)

또한, 일반적인 관행에 따라 training을 2개의 phase로 분리

1. data space에 perceptual하게 동등한 low-dimensional representation space를 제공하는 autoencoder 훈련
2. spatial dimensionality와 관련해 scaling property를 더 잘 나타내는 learning latent space에서 DM을 훈련
    
    ⇒ 과도한 공간 압축 의존 필요 X
    

복잡성 감소 → single network pass로 latent space에서 효율적인 image generation 제공

그 결과 모델 클래스 ⇒ LDMs(Latent Diffusion Models)

**Contriutes**

1. purely transformer-based approach와 달리 고차원 데이터에 더 우아하게 확장
⇒ 기존 방법들보다 더 믿을 수 있고 디테일한 reconstruction을 제공하는 compression level에서 작업할 수 있고, mega pixel image의 high-resolution synthesis에 효율적으로 적용될 수 있다. 
2. 계산 비용 및 추론 비용 절감; 여러 task 및 dataset에서 경쟁력 있는 성능
3. reconstruction과 generative ability의 섬세한 weighting이 필요 없음을 보여주며, 이게 매우 믿을 만한 reconstruction을 보장하고 latent space의 regularization을 거의 필요로 하지 않음
4. super-resolution, inpainting 및 semantic synthesis 같은 dense한 조건의 task를 발견
convolution 방식으로 적용 가능; $\sim 1024^2$ px의 크고 일관된 이미지 렌더링 가능
5. DM training 외에도 다양한 task에 사용 가능한 latent diffusion 및 autoencoder model 제공

---

# 2. Comparison with Related Work

### Generative Models for Image Synthesis

---

**GAN (Generative Adversarial Network)**

- good perceptual quality를 갖는 high-resolution image를 효율적으로 샘플링할 수 있지만, 최적화하는 데에 어려움이 있고 충분한 data distribution을 포착해내는 데에도 분투한다.

**Likelihood-based method**

- optimization을 더 잘 수행하도록 만드는 good density estimation을 강조

**VAE (Variational Autoencoders) & Flow-based model**

- 효율적으로 high resolution image를 합성할 수 있지만, 샘플의 품질은 GAN과 동등하지 않음

**ARM (Autoregressive model)**

- density estimation에서 강력한 성능을 달성하지만 계산이 까다로운 아키텍처와 순차적 샘플링 프로세스가 저해상도 이미지로 제한함

Pixel-based representation of image는 high-frequency detail이 거의 인식할 수 없는 수준이기 때문에, maximum-likelihood training은 모델링 하는 데에 불균형한 양의 용량을 소비해 학습 시간이 길어진다. 더 높은 resolution으로 확장하기 위해, 다양한 two-stage approach에서 ARM을 사용해 raw pixel 대신 **compressed latent image space**를 모델링한다. 

### Diffusion Probabilistic Models (DM)

---

DM은 sample quality 뿐만 아니라 density estimation에서도 SOTA 결과를 달성하고 있다. 

- 이러한 generative power는 백본이 UNet일 때 image-like data의 inductive bias에 대한 자연스러운 적합되는 것에서 비롯
- best synthesis quality : reweighted objective가 훈련에 사용될 때 달성
    - 이때 DM은 손실 압축기에 해당하며 압축 용량과 이미지 quality를 trade 하게 함

이러한 모델이 pixel space에서 evaluation 및 optimization 하면 추론 속도가 저하되고 훈련 비용이 높아진다. 

- 추론 속도 저하는 개선된 샘플링 전략과 계층적 접근법으로 해결; but, 여전히 비싼 gradient 계산 요구

⇒ 두 단점 (추론 속도/훈련 비용) 모두 저차원 latent space에서 동작하는 LDM으로 해결 가능

### Two-Stage Image Synthesis

---

**VQ-VAE**

- autoregressive model을 사용해 이산화된 latent space에 대한 expressive prior[표현적인 사전 정보]을 학습

**Zero-shot text-to-image generation**

- 이산화된 image 및 text representation에 대한 joint distribution을 학습하여 text-to-image generation 확장

**Network-to-network translation with conditional invertible neural networks**

- conditionally invertible network를 사용해 다양한 도메인에서 latent space 간 generic transfer를 제공

**VQGANs**

- VQ-VAE와 달리 autoregressive transformer를 더 큰 이미지로 확장하기 위해 adversarial 및 perceptual objective를 갖는 first stage를 사용

그러나 수십억 개의 trainable parameter를 갖는 ARM training에 필요한 높은 압축률이 이러한 접근법들의 성능을 전체적인 성능을 제한하고 높은 계산 비용을 초래한다. 

본 논문의 방법은 LDM이 convolution backbone 때문에 더 부드럽게 고차원 latent space로 확장되는 것과 같은 tradeoff를 방지한다. 

→ 따라서 높은 fidelity reconstruction을 보장하면서 generative DM까지 너무 많은 perceptual compression을 남기지 않고 강력한 첫 번째 stage 사이에서 최적으로 중재하는 compression level을 자유롭게 선택할 수 있다. 

jointly/seperately하게 encoding/decoding 모델을 score-based proor와 함께 학습하는 접근 방식이 존재하지만,

- jointely 한 것은 여전히 reconstruction과 generation 기능 간 어려운 weighting을 필요로 하고,
- seperately 한 것은 사람의 얼굴과 같은 highly structured image에 초점을 맞추고 있다.

---

# 3. Key point

### Simple Concept

---

Autoencoder를 통해 image space와 perceptual하게 동등하지만 연산량은 줄일 수 있는 latent space를 학습하고, 

UNet의 inductive bias인 spatial structure data일 때 자연스럽게 적합되는 강점을 가지며, general-purpose compression model을 얻을 수 있다. 

### 1. Perceptual Image Compression

---

본 논문의 perceptual compression model

- perceptual loss와 patch-based adversarial objective의 조합으로 훈련된 autoencoder로 구성
    
    ⇒ local realism을 적용하여 reconstruction이 image manifold에만 국한되어 $L_2$나 $L_1$ objective와 같은 pixel-space loss에만 의존함으로써 흐릿함이 생기는 것 방지 가능
    
- RGB space의 image $x \in \mathbb R^{H \times W \times 3}$가 주어지면,
    - encoder $\varepsilon$가 $x$를 latent representation $z = \varepsilon(x)$으로 인코딩하고
    - decoder $\mathcal D$가 latent representation으로부터 image를 재구성하여 $\tilde{x} = \mathcal D(z) = \mathcal D( \varepsilon (x))$를 생성
    - 중요한 것은 이 encoder가 image를 factor $f = H/h = W/w$로 다운샘플링한다는 것.
    연구진은 $m \in N$에 대해 서로 다른 다운샘플링 factor $f = 2^m$에 대해 조사했다.

 variance가 높은 latent space를 피하기 위해, 연구진은 두 가지 다른 종류의 regularization으로 실험한다. 

1. KL-reg. 
    - VAE와 비슷하게, learned latent space에서 standard normal에 약간의 KL-penalty 부여
2. VQ-reg.
    - decoder 내에서 vector quantization layer를 사용
    - 디코더에 흡수된 형태로 quantization layer를 가지긴 했지만 VQGAN으로 해석될 수 있음

 본 논문에서 제안하는 DM은 learned space $z = \varepsilon (x)$의 2차원 구조로 동작하도록 설계되기 때문에, $z$가 1차원 구조였던 이전보다 상대적으로 $x$의 detail을 더 잘 보존한다. 

→ 가벼운 압축률; reconstruction도 잘 수행

![Untitled](./image/Untitled%201.png)

### 2. Latent Diffusion Model

---

Diffusion Model이란?

- 정규 분포 변수를 점진적으로 denoising하여 data distribution $p(x)$를 학습하도록 설계된 확률론적 모델로, length T의 fixed Markov Chain의 역과정을 학습하는 것
- 가장 성공적인 모델 특징
    
    ⇒ variational lower bound의 reweighted variant에 의존; denoising score-matching 반영
    
    - input x의 noisy version인 input $x_t$의 denoised variant를 예측하도록 훈련된 autoencoders $\epsilon_\theta(x_t, t); \ t =1,...,T$의 equally weighted sequence로 해석 가능
    - 단순화된 objective; (쉽게 말하면 noisy input $x_t$로부터 original input $x$ reconstruction)
        
        $$
        L_\text{DM} := \mathbb E_{x, \epsilon \sim \mathcal N(0,1), t} \left[ \left\| \epsilon - \epsilon_\theta(x_t, t )\right\|^2_2 \right] \ \ \ \ (1)
        $$
        

Generative Modeling of Latent Representation

- 본 논문의 perceptual compression model을 통해 이제 high-frequency이고 imperceptible detail은 추상화되는 효율적인 저차원 latent space에 접근 가능
    
    this latent space is…
    
    1. data의 중요한, semantic bit에 집중할 수 있고, 
    2. 더 낮은 차원에서 학습할 수 있어 계산적으로 효율적
        
        ⇒ likelihood-based generative model에 적합
        
- 심하게 압축된 discrete latent space에서 autoregressive, attention-based transformer model에 의존하던 이전 연구와 달리, 모델이 제공하는 image-specific inductive bias 활용 가능
- 신경망 백본인 $\epsilon_\theta(o, t)$는 time-conditional UNet.
    - forward process가 고정되어 있어 $z_t$는 학습 과정에서 $\varepsilon$으로 쉽게 구할 수 있고 $p(z)$로부터의 샘플은 $\mathcal D$를 통해 single pass로 쉽게 denoising 가능
        
        $$
        L_\text{LDM} := \mathbb E_{\varepsilon(x), \epsilon \sim \mathcal N(0,1), t} \left[ \left\| \epsilon - \epsilon_\theta(z_t, t )\right\|^2_2 \right] \ \ \ \ (2)
        $$
        

### 3. Conditioning Mechanisms

---

 다른 유형의 generative model과 마찬가지로 diffusion model도 원리적으로 $p(z|y)$ 형태의 conditional distribution을 모델링할 수 있다. 이로 인해 conditional denoising autoencoder $\varepsilon_\theta (z_t, t, y)$로 구현될 수 있고 이것이 바로 synthesis process(ex. text, semantic map, image-to-image translation task)를 컨트롤하는 방법이다. 

 본 논문에서는 다양한 input modality의 attention-based model을 효과적으로 학습할 수 있는, cross-attention mechanism을 사용하는 기본 UNet backbone을 증강하여 DM보다 유연한 conditional image generator를 만든다. 다양한 양식에 대한 pre-process를 위해 intermediate representation $\tau_\theta (y) \in \mathbb R^{M \times d_\tau}$에 $y$를 projection하는 domain specific encoder $\tau_\theta$를 도입하고, 이는 $\text{Attention} (Q, K, V) = \text{softmax} (\frac{QK^T}{\sqrt{d}}) \cdot V$를 구현하는 cross-attention layer 구현을 통해 UNet의 intermediate layer에 매핑된다. 

$$
Q = W^{(i)}_Q \cdot \varphi_i (z_t), \ K = W^{(i)}_K \cdot \tau_\theta(y), \ V = W^{(i)}_V \cdot \tau_\theta(y) \\ W^{(i)}_V \in \mathbb R^{d \times d^i_\epsilon}, \ \ \ W^{(i)}_Q,W^{(i)}_K \in \mathbb R^{d \times d_\tau}, \ \ \ \varphi_i(z_i) \in \mathbb R^{N \times d^i_\epsilon}, \ \ \ \tau_\theta(y) \in \mathbb R^{M \times d_\tau}
$$

- $\varphi_i(z_i) \in \mathbb R^{N \times d^i_\epsilon}$ : $\epsilon_\theta$를 구현하는 UNet의 (flattened) intermediate representation
- $W^{(i)}V \in \mathbb R^{d \times d^i\epsilon}, \ \ \ W^{(i)}_Q,W^{(i)}K \in \mathbb R^{d \times d\tau}$ : 학습 가능한 projection matrix

 image conditioning pair를 기반으로, conditional LDM을 다음과 같이 학습할 수 있다. 

$$
L_\text{LDM} := \mathbb E_{\varepsilon(x), y, \epsilon \sim \mathcal N(0,1), t} \left[ \left\| \epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y) \right\|^2_2 \right] \ \ \ \ (3)
$$

- 위 식을 통해 $\tau_\theta$와 $\epsilon_\theta$가 공동으로 최적화된다.

---

# 4. Algorithm

![Untitled](./image/Untitled%202.png)

---

# 5. Formulation

### 1. Perceptual Image Compression

---

$x \in \mathbb R^{H \times W \times 3}$ : input data

$\varepsilon$ : encoder

$\mathcal D$ : decoder

$z = \varepsilon (x)$ : latent representation

$\tilde{x} = \mathcal D(z) = \mathcal D(\varepsilon (x))$ : reconstruction by decoder

### 2. Latent Diffusion Model

---

$$
L_\text{DM} := \mathbb E_{x, \epsilon \sim \mathcal N(0,1), t} \left[ \left\| \epsilon - \epsilon_\theta(x_t, t )\right\|^2_2 \right] \ \ \ \ (1)
$$

$$
L_\text{LDM} := \mathbb E_{\varepsilon(x), \epsilon \sim \mathcal N(0,1), t} \left[ \left\| \epsilon - \epsilon_\theta(z_t, t )\right\|^2_2 \right] \ \ \ \ (2)
$$

### 3. Conditioning Mechanisms

---

$$
Q = W^{(i)}_Q \cdot \varphi_i (z_t), \ K = W^{(i)}_K \cdot \tau_\theta(y), \ V = W^{(i)}_V \cdot \tau_\theta(y) \\ W^{(i)}_V \in \mathbb R^{d \times d^i_\epsilon}, \ \ \ W^{(i)}_Q,W^{(i)}_K \in \mathbb R^{d \times d_\tau}, \ \ \ \varphi_i(z_i) \in \mathbb R^{N \times d^i_\epsilon}, \ \ \ \tau_\theta(y) \in \mathbb R^{M \times d_\tau}
$$

$$
L_\text{LDM} := \mathbb E_{\varepsilon(x), y, \epsilon \sim \mathcal N(0,1), t} \left[ \left\| \epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y) \right\|^2_2 \right] \ \ \ \ (3)
$$

---

# 6. Insight

### Limitation

---

LDM이 기존 DM보다 계산 비용을 훨씬 감소시킨 것은 사실이지만 여전히 GAN보다 느리다. 또한, 높은 정밀도가 필요할 때는 사용하기 애매할 수 있다. image quality의 loss는 매우 적지만 pixel space에서 세분화된 accuracy가 요구되는 task에 대해서는 reconstruction capability가 bottleneck이 될 수 있다. 연구진은 이러한 관점에서 super-resolution model은 이미 어느 정도 **한계**에 이르렀다고 가정한다. 

---