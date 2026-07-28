# Mixed-Effects Engine v3 — 수학적 참조 문서

Date: 2026-07-28
Scope: 이번 세션에서 최종 확정된 방법론 전체(`MixedEffectsModeling/core/`, `validation/`, `PCIS_Calibration/`)의
이론적 배경·유도·구현을 기록한다. 설계 배경/동기는 `2026-07-27-eb-dispersion-cook-outlier-design.md`를 보라 — 이
문서는 "왜 이걸 택했는가"가 아니라 "이게 수학적으로 정확히 무엇을 계산하는가"에 집중한다.

## 0. 전체 파이프라인

**핵심은 PCIS가 "stage들 다음의 별도 단계"가 아니라, 각 stage의 적합 루틴 그 자체에 내장돼 있다는 것.**
반면 dispersion intercept squeeze는 그 반대로 — 유전자 하나의 stage 적합 안에서 일어나는 게 아니라, **전
유전자 캐스케이드가 끝난 뒤 한 번, 전 유전자를 모아 사후처리**로 적용된다(그래야 $\tau_d$를 안정적으로
추정할 표본이 생긴다, §2.3). 즉 파이프라인은 유전자 루프(캐스케이드+PCIS)와 그 바깥의 전역 루프(squeeze)
두 겹이다:

```
Phase 0   pilot 적합(무prior) → covariate-adjusted lowess trend  α_trend(μ)
             │                                              │
             ├─▶ EB slope prior τ_k (Xγ에 사용)               ├─▶ PCIS의 잔차 척도로 사용 (§4.2)
             │                                              └─▶ intercept squeeze의 목표 μ_g (§2.3)
             ▼
┌─ 유전자마다, 캐스케이드 ───────────────────────────────────────────────┐
│                                                                        │
│  Stage nbi_full_eb                                                    │
│    fit: log(mu)=Xβ+Zb,  log(theta)=Xγ (γ_k에 EB slope prior)          │
│    → PCIS 계산 (α_trend 기준, §4) → 초과분 제거 → 1회 재refit          │
│    성공 ──────────────────────────────────────────┐                   │
│    실패                                            │                   │
│    ▼                                               │                   │
│  Stage nbi_intercept_eb                            │                   │
│    fit: log(mu)=Xβ+Zb,  log(theta)=γ0              │                   │
│    → PCIS 계산 → 초과분 제거 → 1회 재refit          │                   │
│    성공 ────────────────────────────────────────────┤                   │
│    실패 → route = excluded                          │                   │
└──────────────────────────────────────────────────────┼──────────────────┘
                                                         ▼
                              (캐스케이드가 전 유전자에 대해 끝난 뒤, 전역 1회)
                       Dispersion intercept EB squeeze — γ0 → precision-weighted(γ0, trend) (§2.3)
                                                         │
                                                         ▼
                              RQR (randomized quantile residual) → Z-score  (§6)
                                                         │
                                                         ▼
                              SHASH per-gene 재보정 (미변경, §7)
```

정리하면 PCIS는 **stage 내부, γ0 squeeze 이전** — 즉 squeeze되지 않은(raw) dispersion coefficient로 계산된
$\alpha_{\mathrm{trend}}$를 잔차 척도로 쓴다(애초에 PCIS의 α는 그 유전자의 계수가 아니라 Phase 0 trend 함수값
이므로 squeeze 여부와 무관하다). Squeeze는 그 반대로 **PCIS가 이미 다 끝난 뒤**, 이상치가 제거된 상태로
재적합된 disp coefficient들을 대상으로 실행된다.

trend이 EB shrinkage의 목표점이면서 PCIS의 분산 기준이기도 하므로, Phase 0이 정확해야 나머지 전부가 정확해진다 —
이 문서의 절 순서도 그 의존관계를 따른다.

---

## 1. Phase 0 — Covariate-Adjusted Dispersion Trend

### 1.1 무엇을 추정하는가

각 유전자 $g$의 NB2 조건부 분산은 $\mathrm{Var}(y\mid\mu) = \mu + \alpha_g\mu^2$이다. Trend의 역할은
"이 유전자에 대한 정보가 부족할 때, 비슷한 발현 수준의 다른 유전자들은 $\alpha$가 대략 얼마인가"를 answer하는
함수 $\alpha_{\mathrm{trend}}(\bar\mu_g)$를 만드는 것이다.

### 1.2 왜 raw-count MoM은 틀렸는가 (분산분해)

가장 단순한 추정은 공변량·배치를 무시한 원시 카운트의 method-of-moments:
$\hat\sigma_{\mathrm{MoM}} = (\mathrm{Var}(y_g) - \bar y_g)/\bar y_g^2$.

문제는 이 양이 조건부 분산이 아니라는 것이다. $\log\mu_i = \beta_0+\beta'X_i+b_j$인 모형에서 law of total
variance를 전개하면

$$\mathrm{Var}(y) = \underbrace{E[\mu]}_{\text{Poisson}} + \underbrace{\alpha\,E[\mu^2]}_{\text{NB overdispersion}} + \underbrace{\mathrm{Var}_j(\text{batch effect})}_{\tau^2} + \underbrace{\mathrm{Var}_X(\exp(\beta'X))}_{\text{공변량이 만드는 평균의 변동}}$$

이고 정규화하면 근사적으로

$$\hat\sigma_{\mathrm{MoM}} \approx \alpha + \tau^2 + \mathrm{CV}^2_X\!\big(\exp(\beta'X)\big)$$

즉 raw MoM은 진짜 dispersion $\alpha$뿐 아니라 배치 분산과 **평균 submodel이 이미 설명하는 공변량 변동까지
전부 합산**한다. 실측(19,085개 적합 유전자 기준): $\bar\mu\approx 819$에서 세 항의 합 1.541 대 관측 1.495
(alpha 7.1% / $\tau^2$ 5.7% / 공변량 82.3%), $\bar\mu\approx 0.19$에서 alpha 45.5% / 공변량 37.0%. 공변량
항의 비중이 발현이 높을수록 급격히 커지므로, raw trend는 고발현 구간에서 특히 심하게 과대추정한다(2.30배 →
16.71배, 단조 증가).

### 1.3 해법 — pilot 적합의 조건부 dispersion에 lowess

`build_trend_from_fits`는 raw count가 아니라 **pilot 단계에서 이미 공변량으로 조건화하여 적합된**
$\hat\alpha_g$(stage `nbi_full_eb`, dispersion prior 없이) 위에 lowess를 적용한다. 이는 edgeR/DESeq2의
trended dispersion과 동일한 연산 순서다: 먼저 조건부 모형으로 평균을 설명하게 하고, 남은 dispersion만
발현수준의 매끄러운 함수로 요약한다.

### 1.4 LOWESS의 수학 (Cleveland 1979)

목표 함수: $x = \log\bar\mu_g$, $y=\log\hat\alpha_g$ ($G=1{,}736$개 pilot 유전자). 각 평가점 $x_0$마다
**국소 선형 가중회귀**를 다시 푼다.

**(a) 이웃 폭과 가중치.** 최근접 이웃 개수 $q = \lfloor \texttt{frac}\cdot G\rfloor$ (frac=0.3),
$d(x_0) = $ $x_0$에서 $q$번째로 가까운 점까지의 거리. Tricube 커널:

$$K_i(x_0) = \Big(1-\big(|x_i-x_0|/d(x_0)\big)^3\Big)^3_+$$

윈도 밖($|x_i-x_0|>d(x_0)$)은 가중치 0 — 즉 적응적(국소 밀도에 따라 폭이 변하는) 유한 지지 커널이다.

**(b) 국소 선형 WLS의 닫힌 해.** 국소좌표 $t_i = x_i-x_0$에서

$$\min_{b_0,b_1}\ \sum_i K_i\,(y_i-b_0-b_1 t_i)^2$$

정규방정식 $\begin{pmatrix}S_0&S_1\\S_1&S_2\end{pmatrix}\begin{pmatrix}b_0\\b_1\end{pmatrix}=\begin{pmatrix}T_0\\T_1\end{pmatrix}$,
$S_k=\sum_i K_i t_i^k,\ T_k=\sum_i K_i t_i^k y_i$. Cramer's rule로

$$\hat y(x_0) = b_0 = \frac{S_2 T_0 - S_1 T_1}{S_0 S_2 - S_1^2}$$

($t_i$가 $x_0$ 기준 국소좌표이므로 $x_0$에서의 예측값은 절편 $b_0$ 그 자체.)

**(c) Robustifying iteration (bisquare).** 이상치(개별 유전자의 dispersion 추정 노이즈)에 대한 강건성을
위해, (b)에서 나온 잔차 $e_i = y_i-\hat y(x_i)$로 재가중한다: $s=\mathrm{median}|e_i|$,

$$\delta_i = \Big(1-\big(e_i/(6s)\big)^2\Big)^2_+$$

다음 반복에서 가중치를 $K_i\delta_i$로 바꿔 (b)를 다시 푼다 (it=3회). 큰 잔차를 가진 점(=국소적으로 이상한
유전자)의 영향력을 점진적으로 죽인다 — 이게 여기서 "robust"의 의미이며, 임계값 함수가 아니라 **가중치가
연속적으로 줄어드는 것**이 핵심이다.

전체 $O(G^2)$이지만 $G=1{,}736$이라 무시할 수준이고, 슬로프 추정 자체(캐시된 pilot 적합)가 병목이다.

### 1.5 검증

Bias: 유전자별 |median residual| $\le 0.067$/bin, 전체 1.520 → 0.008. 다운스트림 효과: $\tau_d^2$ 0.460 →
**0.115**(현재 실행 기준 `eb_meta.json`: 0.131), squeeze가 3–5배 강해짐; PCIS가 고발현 구간에서 검출력을
회복(§4.2).

---

## 2. Empirical-Bayes Dispersion Shrinkage

### 2.1 계층모형과 moment estimator (limma/edgeR/DerSimonian-Laird 계열)

각 유전자의 dispersion(로그스케일) MLE $\hat\phi_g$가

$$\hat\phi_g \mid \phi_g \sim (\phi_g,\ SE_g^2),\qquad \phi_g \sim (\mu_g,\ \tau^2)$$

를 만족한다고 하자($\mu_g$는 slope의 경우 0, intercept의 경우 그 유전자의 trend값). 그러면

$$\mathrm{Var}(\hat\phi_g - \mu_g) = \tau^2 + E[SE_g^2]$$

이 method-of-moments 항등식(랜덤효과 메타분석의 DerSimonian–Laird 추정량과 같은 구조, limma의
`squeezeVar`·edgeR의 `estimateGLMRobustDisp`와 동일 계열)에서 좌변을 관측된 잔차의 robust variance로,
우변 두번째 항을 표본 SE들의 typical value로 바꿔 추정한다:

$$\hat\tau^2 = \max\!\Big(0,\ \big(1.4826\cdot\mathrm{MAD}_g(\hat\phi_g-\mu_g)\big)^2 - \mathrm{median}_g(SE_g^2)\Big)$$

MAD·median(평균·분산 아님)을 쓰는 이유: 일부 유전자가 거의 발산에 가까운 MLE를 낼 수 있는데, 그런 소수가
sample variance/mean을 왜곡시켜 $\tau$가 인위적으로 커지고(=수축이 거의 꺼짐) 하는 걸 막기 위함. $1.4826$은
정규분포에서 $\mathrm{MAD}\to\sigma$로 바꾸는 표준 상수($1/\Phi^{-1}(0.75)$).

### 2.2 Posterior mean (precision-weighted shrinkage)의 유도

정규-정규 켤레: likelihood $\hat\phi_g\mid\phi_g\sim N(\phi_g,SE_g^2)$, prior $\phi_g\sim N(\mu_g,\tau^2)$.
로그밀도의 $\phi_g$에 대한 이차형식을 완전제곱화하면

$$-\tfrac12\Big[\tfrac{(\hat\phi_g-\phi_g)^2}{SE_g^2}+\tfrac{(\phi_g-\mu_g)^2}{\tau^2}\Big]
\ \propto\ -\tfrac12\Big(\tfrac1{SE_g^2}+\tfrac1{\tau^2}\Big)\Big(\phi_g-\underbrace{\tfrac{\hat\phi_g/SE_g^2+\mu_g/\tau^2}{1/SE_g^2+1/\tau^2}}_{\text{posterior mean}}\Big)^2$$

즉

$$\phi_g\mid\hat\phi_g \sim N\!\Big(\underbrace{\frac{\hat\phi_g/SE_g^2+\mu_g/\tau^2}{1/SE_g^2+1/\tau^2}}_{\text{precision-weighted 평균}},\ \big(1/SE_g^2+1/\tau^2\big)^{-1}\Big)$$

이게 코드의 `squeeze_log_theta`가 계산하는 값이다. $SE_g\to 0$(그 유전자의 증거가 강함)이면 posterior mean
$\to\hat\phi_g$(수축 없음); $SE_g\to\infty$(증거 없음/NaN)면 $\to\mu_g$(trend가 곧 답).

### 2.3 두 갈래 적용

**(a) Dispersion slopes** ($\gamma_1,\dots,\gamma_{10}$, 공변량별). Prior 평균 $\mu_g=0$(귀무: 공변량이
dispersion에 영향 없음)으로 두고, `--mode pilot`(dispersion prior 없이 `nbi_full_eb` 적합, HC 평균발현
10분위 층화표집 2,000유전자 목표, 실제 1,736개 수렴)에서 나온 계수 스프레드로 공변량별 $\tau_k$를 추정:

$$\tau_k = \sqrt{\max\big(0,\ (1.4826\,\mathrm{MAD}_g(\hat\gamma_{gk}))^2 - \mathrm{median}_g(SE_{gk}^2)\big)}$$

실측값(`disp_prior.json`, 이번 실행): 0.144(rRNA Fraction) ~ 0.396(Gene Length Bias), 전 구간 v2의 고정값
0.05보다 3~8배 넓다. 본 실행에서는 이 $\tau_k$를 `normal(0,\tau_k)` prior로 실제 캐스케이드에 투입한다
(glmmTMB `priors=` API, `class="betad"`).

**(b) Dispersion intercept** $\gamma_0$. Prior에 넣지 않고(넣으면 §2.1의 $SE_g$ 추정 자체가 오염) 적합
후 **한 번의 analytic squeeze**를 적용한다. 목표값 $\mu_g = -\log\alpha_{\mathrm{trend}}(\bar\mu_g)$(Phase
0에서 나온 값), stage `nbi_full_eb`+`nbi_intercept_eb`를 풀링해 하나의 $\tau_d$ 추정(어느 한쪽만으로는
표본이 부족):

$$\hat\tau_d^2 = 0.131\ (\text{현재 실행 기준}),\qquad \tau_d = 0.363$$

$SE_g=\mathrm{NaN}$(glmmTMB `sdreport` 실패)이면 $SE_g^2=\infty$로 처리 — §2.2 공식에서 이는 정확히
$\gamma_0\to\mu_g$, 즉 v2의 "trend에 완전히 고정" 정책이 EB 규칙의 극한 케이스로 자연스럽게 복원된다.
수축 강도(shrink weight) $w_g = SE_g^2/(SE_g^2+\tau_d^2)$는 실측으로 잘 측정된 유전자에서 0.006~0.14 수준
— 즉 대다수 유전자는 자기 데이터가 trend보다 훨씬 강한 증거이므로 거의 수축되지 않는다.

*근사의 한계:* 평균 계수·$\tau^2$는 squeeze된 dispersion 하에서 재적합되지 않는다(joint posterior mode가
아님). limma/edgeR도 동일한 근사를 쓰며, RQR의 Z-값 보정에 실제로 관여하는 건 $\alpha$이므로 이 근사가
허용된다.

---

## 3. 2단계 캐스케이드

| stage | 평균 submodel | dispersion submodel |
|---|---|---|
| `nbi_full_eb` | $\log\mu_i=\beta_0+\beta'X_i+b_j,\ b_j\sim N(0,\tau^2)$ | $\log\theta_i=\gamma_0+\gamma'X_i$, 슬로프 EB prior, 절편 squeeze |
| `nbi_intercept_eb` | 동일 | $\log\theta_i=\gamma_0$, 절편 squeeze |

두 stage 모두 같은 EB 절편 squeeze를 공유하므로 이름이 같다(`_eb`). 둘 다 실패한 유전자는
`route="excluded"`. v2의 3단계 중 `nbi_disp_intercept`가 사라진 이유: 슬로프 prior가 정확히 캘리브레이션된
지금, 그 stage는 `nbi_full_eb`와 구조적으로 거의 구분되지 않는다(0.05로 과수축됐던 v2에서만 별개 stage처럼
보였을 뿐).

---

## 4. PCIS (Prior-Conditioned Impact Score)

### 4.1 출발점 — GLM의 one-step Cook's distance (Pregibon 1981)

선형모형의 Cook's distance는 "관측 $i$를 실제로 빼고 다시 적합했을 때 계수가 얼마나 움직이는가"의
정의이다:

$$D_i = \frac{(\hat\beta-\hat\beta_{(i)})'(X'WX)(\hat\beta-\hat\beta_{(i)})}{p\,\hat\phi}$$

실제 재적합은 비싸므로, IRLS 수렴점에서의 **가중최소제곱 leave-one-out 공식**(Sherman–Morrison류)으로 근사한다:

$$\hat\beta_{(i)} \approx \hat\beta - \frac{(X'WX)^{-1}x_i\, w_i e_i}{1-h_{ii}},\qquad h_{ii}=w_i\,x_i'(X'WX)^{-1}x_i$$

대입하면 $(\hat\beta-\hat\beta_{(i)})'(X'WX)(\hat\beta-\hat\beta_{(i)}) = w_i e_i^2\,h_{ii}/(1-h_{ii})^2$이고,
$r_i^2 \equiv w_i e_i^2$(표준화 잔차 제곱)로 다시 쓰면 익숙한 형태가 나온다:

$$D_i \approx \frac{r_i^2}{p\,\hat\phi}\cdot\frac{h_{ii}}{(1-h_{ii})^2}$$

이게 "실제 재적합 없이 삭제 효과를 근사"하는 표준 one-step 근사이고, $\hat\phi$가 그 모형 자신의 fitted
dispersion, $h_{ii}$가 fitted 설계행렬의 leverage라는 전제 위에서만 근사가 성립한다. PCIS는 이 두 전제를
둘 다 의도적으로 깬다.

### 4.2 이탈 1 — $\hat\phi \to \alpha_{\mathrm{trend}}$ (self-masking 차단)

NB2의 dispersion MLE는 잔차 크기(대략 deviance 형태의 항)에 의해 직접 움직인다: 극단적으로 큰 $y_i$ 하나가
있으면 그 우도를 "정상적인 큰 분산"으로 설명하는 쪽으로 $\hat\alpha$가 밀려 올라간다. $D_i \propto 1/\hat\phi$이므로
이는 정확히 §Externally-studentized-residual 논의에서 다룬 self-masking이다(OLS에서 $e_i$가 자기 자신의
$\hat\sigma^2$를 부풀려 스스로를 가리는 것과 동일 메커니즘, Belsley–Kuh–Welsch 1980; Cook–Weisberg 1982).
실측: 합성 3개 20배 이상치가 $\hat\alpha$를 0.004→0.147(36배)로 부풀려 통계량을 4.5–9.4에서 0.6–1.1로 깎아
**모든 이상치 강도에서 0건 검출**.

해법은 §1에서 만든 $\alpha_{\mathrm{trend}}(\bar\mu_g)$로 $\hat\phi$를 대체하는 것 — "이 유전자 자신을
제외한" 정보(비슷한 발현대의 ~1,900개 다른 유전자)로 척도를 추정한다는 점에서 externally studentized
residual과 같은 원리이지만, 단위가 "관측 1개 제외"가 아니라 "유전자 전체 제외"라서 다중 이상치(같은
유전자 안의 여러 오염 관측)에도 훨씬 강건하다 — 유전자 하나 안의 이상치 몇 개가 수천 유전자로 만든 trend를
흔들 여지가 없다(단일 삭제 진단이 다중 이상치에서 무력해지는 문제, Hadi & Simonoff 1993; 이를 회피하는
high-breakdown 추정의 일반 논리는 Rousseeuw & Leroy 1987).

### 4.3 이탈 2 — Prior-penalized mixed leverage

$\mu_i$에는 이미 BLUP $b_j$가 들어 있는데($\mu_i = \exp(\beta_0+\beta'X_i+b_j)$), (4.1)의 $h_{ii}$는
고정효과 설계 $X$만으로 계산된다 — 즉 통계량 안에 "고정효과만의 모형"과 "배치 랜덤효과가 있는 모형"이
섞여 있다.

**Henderson의 mixed model equations를 ridge 형태로 다시 쓰기.** $y=X\beta+Zb+\varepsilon$,
$b\sim N(0,\tau^2 I)$, $\varepsilon\sim N(0,W^{-1})$ (NB2 log-link 작업척도)일 때, $(\beta,b)$의 결합
GLS/BLUP 추정은 다음 벌점화 목적함수의 최소화와 동치이다(Robinson 1991, "That BLUP is a Good Thing"):

$$\min_{\beta,b}\ (y-X\beta-Zb)'W(y-X\beta-Zb) + \frac{1}{\tau^2}\,b'b$$

이는 $b$에 능형(ridge) 벌점을 준 GLS이고, $M=[X\ Z]$, $P=\mathrm{blkdiag}(0_p,\ I/\tau^2)$로 쓰면 정규방정식은

$$(M'WM+P)\begin{pmatrix}\hat\beta\\\hat b\end{pmatrix} = M'Wy$$

fitted value는 $\hat y = Hy$, $H = M(M'WM+P)^{-1}M'W$ — **능형회귀의 hat matrix**와 정확히 같은 형태이며,
$\mathrm{tr}(H)=p_{\mathrm{eff}}$가 유효자유도(ridge/smoother의 effective df 개념, Hastie–Tibshirani–Friedman;
mixed model에 대한 동일 논리는 Hodges & Sargent 2001 "who's in charge" effective df)가 된다.

PCIS는 이 $h_{ii}=H_{ii}$, $p_{\mathrm{eff}}=\mathrm{tr}(H)$를 §4.1의 $h_{ii},p$ 자리에 그대로 대입한다:

$$w_i = \frac{\mu_i}{1+\alpha_{\mathrm{trend}}\mu_i}\ (\text{NB2 log-link IRLS weight}),\qquad
r_i = \frac{y_i-\mu_i}{\sqrt{\mu_i+\alpha_{\mathrm{trend}}\mu_i^2}}$$

$$\mathrm{PCIS}_i = \frac{r_i^2}{p_{\mathrm{eff}}}\cdot\frac{h_{ii}}{(1-h_{ii})^2}$$

$\tau^2\to 0$(30%의 유전자에서 발생하는 배치분산 붕괴)이면 벌점 $1/\tau^2\to\infty$가 되어 $b$가 완전히
억제되고 $p_{\mathrm{eff}}\to p$로 자동 수렴 — 별도 분기 없이 고정효과 전용 모형의 특수 케이스가 저절로
나온다. 실측: $p_{\mathrm{eff}}=18.4$ 대 $p=11$(모형 복잡도의 40%가 기존 방식에서 무시되고 있었음),
싱글턴 batch의 leverage 0.017→0.165(10배).

### 4.4 왜 F분포를 못 쓰는가

(4.1)의 근사가 성립하려면 통계량의 분산이 **그 모형 자신의** fitted dispersion·leverage여야
$D_i/p \cdot \hat\phi(1-h_{ii}) \approx F_{p,n-p}$ 근사(정규성+point 하나의 영향이 작다는 가정 하의 표준
결과)가 성립한다. PCIS는 §4.2·4.3에서 이 전제를 의도적으로 다른 것(trend, ridge-penalized leverage)으로
바꿨으므로, 이 근사가 성립할 이론적 근거가 없다 — DESeq2가 관례로 쓰는 `qf(0.99,p,n-p)`를 그대로 물려받는
건 정당화되지 않는다(§5).

---

## 5. PCIS 임계값 — 경험적 귀무분포 캘리브레이션

### 5.1 Parametric bootstrap null

이론적 참조분포가 없으므로 시뮬레이션으로 대체한다. 수렴한 각 유전자의 적합값 $(\hat\beta,\hat\gamma,\hat\tau^2)$으로

$$u_j \sim N(0,\hat\tau^2),\qquad y_i^\* \sim \mathrm{NB2}\big(\hat\mu_{0,i}\,e^{u_{j(i)}},\ 1/\alpha_i\big)$$

를 실제 설계행렬 위에서 재생성하고, 같은 stage·같은 EB slope prior로 재적합해 PCIS를 다시 계산한다
(`pcis_null.R`, `pcis_vec`가 §4.3 공식을 그대로 재구현). $y^*$에는 정의상 오염이 없으므로, 여기서 어떤
임계값을 넘는 PCIS든 그 초과는 전부 귀무 노이즈다. 19,158유전자 × 693관측 = 13,276,494개 draw
(유전자당 상위 50개 보존, 어느 임계값에서도 그 cap에 도달하지 않음을 확인).

### 5.2 population rate vs "보존된 표본의 분위수" — 흔한 함정

처음 시도한 방식은 보존된 top-50 표본에 직접 `np.quantile`을 적용하는 것이었는데, 이는 심각하게
왜곡된다: 이 표본은 이미 유전자당 693개 중 상위 7.2%로 잘려 있으므로, 그 안에서의 "90th percentile"은
전체 모집단 기준으로는 대략 99.3th percentile에 해당한다. 올바른 방법은 목표를 **전체 모집단(13.28M
관측) 기준 per-observation false-alarm rate**로 고정하고, 정렬된 전체 PCIS 값에서 그 rate에 해당하는
순위의 값을 읽는 것이다:

$$k = \mathrm{round}(\text{rate}\times n_{\text{genes}}\times n_{\text{obs}}),\qquad \text{cut} = p_{(k)}\ (\text{내림차순 } k\text{번째})$$

동시에 `max_removed_per_gene < 50`으로 top-50 cap에 의한 절단 편향이 없는지 매 rate마다 확인한다.

### 5.3 결과 (`PCIS_Calibration/pcis_rate_table.csv`)

| target rate | population %ile | cut | null removed/gene | real removed/gene | null share |
|---|---|---|---|---|---|
| 1e-3 | 99.90% | 0.305 | 0.693 | 0.096 | 7.2x |
| 1.5e-4 | 99.985% | 1.628 | 0.104 | 0.096 | 1.08 (breakeven) |
| **1e-4** | **99.99%** | **2.282** | **0.069** | 0.096 | **0.72** |
| 1e-5 | 99.999% | 18.61 | 0.007 | 0.096 | 0.07 |

`real removed/gene`(=0.096)는 이전 실행에서 옛 `qf(0.99,...)` 임계(중앙값 ~1.98)로 실제 제거된
개수(`training_summary.csv`의 `n_outliers` 평균) — 실측 PCIS 분포 자체는 저장돼 있지 않아 이 표는 한 점에서만
실측과 비교 가능하다(§5.4에서 open item으로 명시). Null-driven 제거량이 실측 수준 아래로 내려가는 지점은
rate 1.5e-4와 1e-4 사이(population percentile 약 99.985~99.99%)이며, **`pcis_cut = 2.28`**(rate 1e-4)을
breakeven보다 여유를 두고 채택했다. 흥미롭게도 이 범위는 옛 `qf(0.99,...)` 컷(~1.98)을 감싸고 있다 —
이론적 근거는 없었지만 우연히 숫자 자체는 캘리브레이션된 값 근처였다.

### 5.4 최종 규칙

$$\text{PCIS}_i > 2.28 \implies \text{관측 } i\text{ 제거 후보 (largest-first, 최대 } \lfloor 0.05n\rfloor\text{개, `droplevels` 후 1회 재적합)}$$

`config.FIT_PARAMS["pcis_cut"] = 2.28`. F분포 quantile이 아니라 위 시뮬레이션에서 읽은 고정 상수다.

---

## 6. RQR (Randomized Quantile Residual) → Z-score

각 stage의 최종 $(\hat\mu_i,\hat\alpha_i)$로 NB2 CDF를 randomized inverse-transform해 $z_i=\Phi^{-1}(u_i)$를
만든다(이산분포이므로 $u_i\sim\mathrm{Unif}(F(y_i-1),F(y_i))$로 랜덤화 — Dunn & Smyth 1996 계열 표준
기법). §2에서 squeeze된 $\hat\alpha$가 여기 직접 들어가므로, EB 보정의 최종 산출물은 이 Z-score의 분포가
얼마나 $N(0,1)$에 가까운가로 검증된다(§CLAUDE.md 1c, `cv_diagnostics`).

---

## 7. SHASH 재보정 (변경 없음)

`core/calibration.py`가 유전자별 held-out HC Z-score에 sinh-arcsinh 분포(Jones & Pewsey 2009)를 적합해
naive skew/kurtosis를 보정한다. 이번 세션의 변경은 여기 들어가는 입력(Z-score 자체)의 품질을 올리는
것이었지, 이 재보정 단계 자체는 건드리지 않았다.

---

## 참고문헌 (방법론 원출처, 서지 상세는 재확인 필요)

- Cleveland, W.S. (1979). *Robust Locally Weighted Regression and Smoothing Scatterplots.* JASA.
- Pregibon, D. (1981). *Logistic Regression Diagnostics.* Annals of Statistics.
- Belsley, D.A., Kuh, E., Welsch, R.E. (1980). *Regression Diagnostics.* Wiley.
- Cook, R.D., Weisberg, S. (1982). *Residuals and Influence in Regression.*
- Hadi, A.S., Simonoff, J.S. (1993). *Procedures for the Identification of Multiple Outliers in Linear Models.* JASA.
- Rousseeuw, P.J., Leroy, A.M. (1987). *Robust Regression and Outlier Detection.* Wiley.
- DerSimonian, R., Laird, N. (1986). *Meta-Analysis in Clinical Trials.* Controlled Clinical Trials. (moment 기반 tau^2 추정의 원형)
- Smyth, G.K. (2004). *Linear Models and Empirical Bayes Methods for limma.* SAGMB. (squeezeVar과 동일한 precision-weighted shrinkage)
- Robinson, G.K. (1991). *That BLUP is a Good Thing: The Estimation of Random Effects.* Statistical Science. (BLUP = ridge-penalized GLS)
- Hodges, J.S., Sargent, D.J. (2001). *Counting Degrees of Freedom in Hierarchical and Other Richly-Parameterised Models.* Biometrika.
- Hastie, T., Tibshirani, R., Friedman, J. *The Elements of Statistical Learning.* (ridge hat matrix, effective df)
- Jones, M.C., Pewsey, A. (2009). *Sinh-Arcsinh Distributions.* Biometrika.
