# Normative Modeling 엔진 v3 — 핵심 수식

Methods 작성용 수식 정리. 각 항목 뒤에 구현 위치를 적어둔다.
서술적 근거·측정 기록은 [CLAUDE.md](CLAUDE.md), 문헌 계보는
[../EDA/normative_modeling_literature.md](../EDA/normative_modeling_literature.md).

표기: 유전자 $g$, 샘플 $i$, 배치 $j$, 기술 공변량 $X_{ik}$ ($k=1..10$, `config.BIAS_COLUMNS`).
NB2 모수화에서 $\alpha = 1/\theta$ 이고 $\operatorname{Var}(y) = \mu + \alpha\mu^2$.

---

## 1. 규범 모델 (per-gene NB2 GLMM)

핵심은 **평균과 분산을 모두 공변량의 함수로 둔다**는 것이다. 뇌영상 NM에서
GAMLSS가 $\mu,\sigma$를 동시에 모델링하는 것(Bethlehem 2022 brain charts)과 같은 구조이며,
count 데이터라 정규분포 대신 NB2를 쓴다.

$$y_{ig} \mid b_j \;\sim\; \mathrm{NB2}\!\left(\mu_{ig},\, \alpha_{ig}\right)$$

$$\log \mu_{ig} \;=\; \beta_{0g} + \sum_{k=1}^{10}\beta_{kg}X_{ik} \;+\; b_{jg},
\qquad b_{jg}\sim\mathcal{N}(0,\tau^2_g)$$

$$\log\theta_{ig} \;=\; \gamma_{0g} + \sum_{k=1}^{10}\gamma_{kg}X_{ik}
\qquad (\text{stage 1: } \texttt{nbi\_full\_eb})$$

stage 1 실패 시 분산 submodel을 절편만으로 축약한다:

$$\log\theta_{ig} \;=\; \gamma_{0g}
\qquad (\text{stage 2: } \texttt{nbi\_intercept\_eb})$$

둘 다 실패하면 `route="excluded"`.

**배치는 평균 submodel에만 진입한다.** 분산에 넣지 않는 이유: HC의 singleton batch 5개가
각각 자유 파라미터를 얻어, trend가 측정하려는 바로 그 dispersion을 깎아내린다.

구현: `core/glmm_helpers.R`, `core/glmm_fit.R --mode cascade`

---

## 2. Dispersion trend

절편 shrinkage의 prior 평균이자 PCIS의 분산 기준. 캘리브레이션 적합의
**공변량 보정된** dispersion에 lowess를 태운다.

$$\log\hat\alpha_g^{\text{trend}} \;=\; \mathrm{lowess}\!\left(\log\hat\alpha_g \;\sim\; \log\bar\mu_g;\ \text{frac}=0.3,\ \text{it}=3\right)$$

raw-count MoM 방식(`build_trend`)은 공변량·배치를 무시해 진짜 dispersion과 공변량 유래
평균 변동을 뒤섞고 고발현 구간에서 심하게 과대추정하므로 진단용으로만 남긴다.

구현: `core/dispersion_trend.py:build_trend_from_fits`

---

## 3. Empirical Bayes shrinkage

### 3.1 사전분산 모멘트 추정 (limma/edgeR 분해)

$$\tau^2 \;=\; \max\!\left(0,\;\left(1.4826 \cdot \mathrm{MAD}(\hat\phi)\right)^2 \;-\; \mathrm{median}(\mathrm{SE}^2)\right)$$

분산/평균이 아니라 **MAD/median**을 쓴다. 소수의 발산 직전 유전자가 $\tau^2$를 부풀려
shrinkage를 조용히 무력화하는 것을 막기 위해서다.

### 3.2 Dispersion 기울기 prior

$$\gamma_{kg} \;\sim\; \mathcal{N}(0,\ \tau_k^2)$$

$\tau_k$는 층화 표본에서 prior 없이 적합한 뒤 유전자 간 기울기 산포로 읽는다
(`--mode calib` → `disp_prior.json`). 측정값 0.10–0.36.

### 3.3 Dispersion 절편 squeeze (cascade 종료 후 전역 1회)

$$\log\theta_{0g}^{\text{post}}
\;=\;
\frac{\dfrac{\log\hat\theta_{0g}}{\mathrm{SE}_{0g}^{2}} \;+\; \dfrac{\log\theta_g^{\text{trend}}}{\tau_d^{2}}}
     {\dfrac{1}{\mathrm{SE}_{0g}^{2}} \;+\; \dfrac{1}{\tau_d^{2}}}$$

정밀도 가중 평균이다. $\mathrm{SE}_{0g}\to\infty$ (적합 실패)이면 squeeze가 trend로 정확히
붕괴하므로, v2의 별도 `nb_fixed` 단계가 이 식의 극한으로 흡수된다.

*근사*: 평균 계수와 $\tau^2$는 squeeze된 dispersion 하에서 재적합하지 않는다
(joint posterior mode가 아님). limma/edgeR과 동일한 타협이며, Z 캘리브레이션을 지배하는 것은
RQR에 들어가는 $\alpha$다.

구현: `core/eb_shrinkage.py`

---

## 4. PCIS (Prior-Conditioned Impact Score)

Cook's distance와 **의도적으로 다른** 이상치 통계량.

$$w_i \;=\; \frac{\mu_i}{1+\alpha^{\text{trend}}\mu_i}
\qquad\text{(NB2 log-link IRLS 가중치)}$$

$$M \;=\; [\,X_f \;\; Z\,],
\qquad
P \;=\; \mathrm{blkdiag}\!\left(0_p,\ I/\tau^2\right)$$

$$H \;=\; W^{1/2} M \left(M^\top W M + P\right)^{-1} M^\top W^{1/2},
\qquad p_{\text{eff}} \;=\; \operatorname{tr}(H)$$

$$r_i \;=\; \frac{y_i-\mu_i}{\sqrt{\mu_i + \alpha^{\text{trend}}\mu_i^2}}$$

$$\boxed{\ \mathrm{PCIS}_i \;=\; \frac{r_i^2}{p_{\text{eff}}}\cdot\frac{h_{ii}}{(1-h_{ii})^2}\ }$$

Cook's D와의 두 가지 차이, 둘 다 측정에 의해 강제된 것:

1. **분산이 그 유전자 자신의 적합값이 아니라 trend $\alpha$다.** 자유 추정된 dispersion은
   20배 규모 이상치가 자기 유전자의 $\alpha$를 부풀려 스스로를 가리게 한다(self-masking).
   trend $\alpha$는 약 19,000개 다른 유전자에서 온 "외부" 값이라, 한 유전자 안의 **동시
   다중 이상치**에도 견딘다 — 관측치 삭제 기반 진단(ESR, Cook's D)이 못 하는 일이다.
2. **leverage를 prior-penalized 혼합 설계에서 계산한다.** $\mu$가 이미 BLUP을 포함하므로
   $X$만으로 hat 행렬을 만들면 유효 복잡도의 약 40%를 무시하게 된다. $\tau^2\to0$이면
   벌점이 무한대가 되어 $p_{\text{eff}}\to p$로 자동 수렴한다.

분산이 적합 모델 자신의 것이 아니므로 **PCIS에는 F 참조분포가 없다.** 절단값은 경험적 null에서
읽은 고정 상수 $\mathrm{PCIS}_{\text{cut}} = 2.28$ (관측치당 population false-alarm rate $10^{-4}$ 목표).
초과 관측치는 큰 것부터 최대 $\lfloor 0.05n \rfloor$개까지 **제거**하고 해당 stage를 1회 재적합한다.

구현: `core/glmm_helpers.R:pcis_outliers`, 캘리브레이션 `core/pcis_null.R` + `PCIS_Calibration/`

---

## 5. Z-score — 이 연구의 실제 산출물

배치 임의효과를 **주변화**한 뒤 randomized quantile residual을 취한다.
이 marginalization 덕분에 **훈련에서 본 적 없는 배치의 샘플도 채점 가능**하다
(BLUP 불필요). CV / LOBO / 질환 코호트 채점이 모두 여기에 의존한다.

Gauss–Hermite($m=7$) 노드 $\{v_m\}$, 정규화 가중치 $\{w_m\}$에 대해

$$F_g^{\text{marg}}(y) \;=\; \sum_{m=1}^{7} w_m \; F_{\mathrm{NB}}\!\left(y \;\middle|\; \mu_{ig}\,e^{\tau_g v_m},\ \alpha_{ig}\right)$$

$$u_{ig} \;\sim\; \mathrm{Uniform}\!\left(F_g^{\text{marg}}(y_{ig}-1),\ F_g^{\text{marg}}(y_{ig})\right)$$

$$\boxed{\ z_{ig} \;=\; \Phi^{-1}(u_{ig})\ }$$

이산 분포이므로 CDF 구간에서 균등 추출하는 **randomized** quantile residual을 쓴다
(Dunn & Smyth 1996). 그래야 모델이 맞을 때 $z \sim \mathcal{N}(0,1)$이 정확히 성립한다.

$\tau^2 < 10^{-6}$인 유전자(약 30%)는 GH를 건너뛰고 직접 NB CDF를 쓴다.

참고로 주변 적률은

$$\mathbb{E}[y] = \mu e^{\tau^2/2},
\qquad
\mathbb{E}[y^2] = \mu e^{\tau^2/2} + (1+\alpha)\mu^2 e^{2\tau^2}$$

$\tau^2$가 크면 $\operatorname{Var}$는 $e^{2\tau^2}$로 폭발하지만 분위수는 거의 움직이지 않는다 —
PPC 분산 패널의 이상 신호는 **적률 아티팩트이지 miscalibration이 아니다.**
$\tau^2$가 실제로 비용을 치르는 곳은 검정력이다 ($\tau^2_{\max}=3.0$의 근거).

구현: `core/marginal_rqr.py:marginal_nb_rqr`

---

## 6. Per-gene SHASH 재캘리브레이션

held-out HC의 Z가 $\mathcal{N}(0,1)$이어야 downstream FDR이 성립한다. RQR 분포에 실제
왜도/첨도가 남는 유전자를 sinh-arcsinh 변환(Jones & Pewsey 2009)으로 교정한다.

$$z^{\text{corr}} \;=\; \sinh\!\left(\delta \,\operatorname{arcsinh}\!\left(\frac{z-\xi}{\eta}\right) - \varepsilon\right)$$

역방향(분위수):

$$Q(p) \;=\; \xi + \eta\,\sinh\!\left(\frac{\operatorname{arcsinh}(\Phi^{-1}(p)) + \varepsilon}{\delta}\right)$$

$(\xi,\eta,\varepsilon,\delta)$는 유전자별 MLE. $\varepsilon$이 왜도, $\delta$가 꼬리 두께를 담당한다.
held-out HC는 **진짜 null**이므로, naive $\mathcal{N}(0,1)$ 가정 대비 BH-FDR 기각률 변화가
위양성 팽창을 직접 측정한다 (Fraza et al. 2021; Efron 2007).

구현: `core/shash.py`, `core/calibration.py:gene_shash_calibration`

---

## 7. Pooled GLMM (저발현 유전자)

비영 관측이 심하게 부족한 유전자($n_z \le 25$)는 개별 적합이 수렴하지 않는다. 이들을 하나의
텐서로 쌓아 고정효과 $\beta$와 배치 분산 $\sigma^2_{\text{batch}}$를 그룹 수준에서 공유시킨다.
유전자별 기저 발현은 offset으로 흡수한다.

$$\log \mu_{i,g} \;=\; \underbrace{\log\!\left(\bar{Y}_{g,\mathrm{HC}} + \epsilon\right)}_{\text{offset}}
\;+\; \beta_0 + \sum_{k=1}^{10}\beta_k X_{ik} \;+\; b_j$$

Poisson으로 먼저 적합하고, Pearson 잔차 과분산비가 임계값(예: 2.0)을 넘으면
$\log\theta=\gamma_0$인 NB2로 재적합한다.

$n_z$ 임계값 25는 개별 cascade 전수 실행에서 측정한 값이다. 수렴 실패율은
$n_z$ 1–5에서 92%, 21–25에서 10%, 31–40에서 4.5%로 knee가 20–25에 있다. 임계값을 더 올리면
**개별 적합이 잘 되는 유전자를 강등**시켜 자기 $\beta$를 공유 $\beta$에 뺏기는데, 그것이
per-gene 규범 모델링의 전제 자체를 무너뜨린다 (구제/강등 비: $n_z$ 25에서 1.21, 30에서 0.97).

**미검증**: `train_pool` / `glmm_fit_pool.R`은 실제 HC 데이터에서 아직 실행된 적이 없다.
CV가 pool 경로의 held-out Z 캘리브레이션을 확인하기 전까지 이 임계값 논증은 조건부다.

구현: `core/glmm_fit_pool.R`

---

## 8. 요약 — 기존 그룹 비교와의 대비

| | 기존 group-wise DE | 본 연구 |
|---|---|---|
| 추정 대상(estimand) | 두 그룹 평균의 차이 $\beta_{\text{group}}$ | 개인의 조건부 편차 $z_{ig}$ |
| 공변량 처리 | 설계행렬의 항 하나 / 사전 제거 | 평균·분산 submodel 양쪽에 내재 |
| 분석 단위 | 코호트 | 샘플 1개 |
| 새 배치 | 재적합 또는 재보정 필요 | GH 주변화로 그대로 채점 |
| 출력 | 유의 유전자 목록 | 유전자 × 샘플 Z 행렬 |
